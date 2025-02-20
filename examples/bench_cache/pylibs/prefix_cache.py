from collections import defaultdict
from typing import List, Tuple
import time
import torch


class TreeNode:
    def __init__(self):
        self.children = defaultdict(TreeNode)
        self.parent :TreeNode = None
        self.key = None
        self.value = None
        self.lock_ref = 0
        self.last_access_time = time.time()

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time

def _key_match(key0: List, key1: List):
    i = 0
    for k0, k1 in zip(key0, key1):
        if k0 != k1:
            break
        i += 1
    return i

class PrefixCache:
    def __init__(self,disable=False):
        # self.root = TreeNode()
        # self.cache = []
        self.disable = disable
        self.reset()
        
        
    def reset(self):
        self.root_node = TreeNode()
        self.root_node.key = []
        self.root_node.value = []
        self.root_node.lock_ref = 1
        self.evictable_size_ = 0


    def _split_node(self, key, child: TreeNode, split_len: int, is_tensor=True):
        # 创建新节点代替child
        new_child_node = TreeNode()
        new_child_node.parent = child.parent
        new_child_node.lock_ref = child.lock_ref
        new_child_node.key = child.key[:split_len]
        new_child_node_value =[]
        splited_value =[]
        if is_tensor:
            for i in range(len(child.value)):
                new_child_node_value.append([child.value[i][0][:,:,:split_len,:],child.value[i][1][:,:,:split_len,:]])
                splited_value.append([child.value[i][0][:,:,split_len:,:],child.value[i][1][:,:,split_len:,:]])
        new_child_node.value = new_child_node_value
        print("new_child_node_key:",len(new_child_node.key))
        print("new_child_node_value:",new_child_node_value[0][0].shape)
        
        # 将child的父节点指向新节点
        child.parent = new_child_node
        child.key = child.key[split_len:]  
        child.value = splited_value
        new_child_node.children = {key[split_len]: child}
        new_child_node.parent.children[key[0]] = new_child_node
        return new_child_node

    def _insert_helper(self, node: TreeNode, key: List, saved_key_values,is_tensor=True):
        node.last_access_time = time.time()
        if len(key) == 0:
            return 0

        if key[0] in node.children.keys():
            child = node.children[key[0]]
            prefix_len = _key_match(child.key, key)

            # 如果prefix_len等于child.key的长度，说明已经匹配到末尾
            if prefix_len == len(child.key):
                # 说明插入节点的key正好被缓存过
                if prefix_len == len(key):
                    return prefix_len
                # 说明插入节点的key的前缀正好被缓存过，需要新的节点缓存后面没有缓存过的内容
                else:
                    key = key[prefix_len:]
                    new_value =[]
                    if is_tensor:   
                        for i in range(len(saved_key_values)):
                            new_value.append([saved_key_values[i][0][:,:,prefix_len:,:],saved_key_values[i][1][:,:,prefix_len:,:]])
                    else:
                        new_value = saved_key_values[prefix_len:]
                    return prefix_len + self._insert_helper(child, key, new_value,is_tensor)
            # 说明插入节点的key的前缀没有被完全缓存过，需要裂变当前节点
            new_child_node = self._split_node(child.key, child, prefix_len,is_tensor)
            new_value =[]
            if is_tensor:
                for i in range(len(saved_key_values)):
                    new_value.append([saved_key_values[i][0][:,:,prefix_len:,:],saved_key_values[i][1][:,:,prefix_len:,:]])
            else:
                new_value = saved_key_values[prefix_len:]
            return prefix_len + self._insert_helper(
                new_child_node, key[prefix_len:], new_value,is_tensor
            )

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = saved_key_values
            node.children[key[0]] = new_node
            self.evictable_size_ += len(saved_key_values)
        return 0

    def insert(self,token_ids:List[int],past_key_values=None,is_tensor=True):
        node = self.root_node
        if past_key_values is None:
            saved_key_values = [x for x in token_ids]
            # return 
        else:
            num_layers = len(past_key_values)
            saved_key_values = []
            for i in range(num_layers):
                saved_key_values.append([past_key_values[i][0],past_key_values[i][1]])
        self._insert_helper(node,token_ids,saved_key_values,is_tensor)
        
        
    def _match_prefix_helper(
        self, node: TreeNode, key: List, last_node: List[TreeNode], is_tensor=True
    ):
        node.last_access_time = time.time()
        if len(key) == 0:
            return
        value = []
        if key[0] in node.children.keys():
            child = node.children[key[0]]
            prefix_len = _key_match(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len,is_tensor)
                value.append(new_node.value)
                last_node.append(new_node)
            else:
                value.append(child.value)
                last_node.append(child)
                match_value,last_node = self._match_prefix_helper(child, key[prefix_len:], last_node, is_tensor)
                if match_value is not []:
                    value += match_value
        return value,last_node
                
    def match(self, key: List[int], is_tensor=True) -> Tuple[torch.Tensor, int]:
            """Find the matching prefix from the radix tree.
            Args:
                key: A list of token IDs to find a matching prefix.
            Returns:
                A tuple of a tensor of matching prefix token IDs and
                the last node that contains the prefix values. Note that
                this API can modify the internal state of the Radix tree.
                The last node create a new child if the prefix is shorter
                than the last node's value.
            """
            if self.disable:
                return None, self.root_node

            # value = []
            last_node = [self.root_node]
            o_value,last_node = self._match_prefix_helper(self.root_node, key, last_node,is_tensor)
            if o_value:
                if is_tensor:
                    final_value = []
                    for layer in range(len(o_value[0])):
                        final_value.append([o_value[0][layer][0].clone(),o_value[0][layer][1].clone()])
                    for i in range(1,len(o_value)):
                        for layer in range(len(o_value[i])):
                            final_value[layer][0] = torch.cat([final_value[layer][0],o_value[i][layer][0]],dim=2)
                            final_value[layer][1] = torch.cat([final_value[layer][1],o_value[i][layer][1]],dim=2)
                   
                    tuple_value = []
                    for layer in range(len(final_value)):
                        tuple_value.append(tuple(final_value[layer]))
                    tuple_value = tuple(tuple_value)
                    value = tuple_value
            else:
                value = None
            # logger.info(f"value: {value}")
            if value is not None:
                # print("query key:",key)
                cached_key = []
                for x in last_node[1:]:
                    cached_key.extend(x.key)
                # print("cached key:",cached_key)
                print("cached key len:",len(cached_key))
                print("query key len:",len(key))
                print("kv shape:",value[0][0].shape)
            return value, last_node[0]

    def _print_helper(self, node: TreeNode, indent: int):
        for _, child in node.children.items():
            print("|"+"_" * indent, len(child.key), child.key[:10], f"r={child.lock_ref}")
            self._print_helper(child, indent=indent + 2)
            
            
# if __name__ == "__main__":
#     cache = PrefixCache()
    
#     cache.insert([1,2,3,4,5,6,7,8,9,10],is_tensor=False)
#     cache.insert([1,2,3,4,5,6,7,8,10,9],is_tensor=False)
#     value,last_node = cache.match([1,2,3,4,5,6,7,8,10,9],is_tensor=False)
#     print(value)
    
    
