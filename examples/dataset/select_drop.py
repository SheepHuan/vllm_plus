import json



kvshare_path ="examples/dataset/data/drop/drop_benchmark_kvshare.json"
fc_path = "examples/dataset/data/drop/drop_benchmark_full_compute.json"
blend_path = "examples/dataset/data/drop/drop_benchmark_cachblend.json"

def check():
    fc = json.load(open(fc_path,"r"))
    blend = json.load(open(blend_path,"r"))
    share = json.load(open(kvshare_path,"r"))
    data1= []
    data2 = []
    for a1,a2,a3 in zip(fc,blend,share):
        # FC对，但是share不对
        if a1["full_compute_score"] == 1.0 and a3["score"] == 0.0:
            data1.append(a3)
        elif a1["full_compute_score"] == 1.0 and a3["score"] == 1.0 and a2["score"]==0.0:
            data2.append(a3)
    json.dump(data1,open("examples/dataset/data/drop/drop_benchmark_fc_yes_share_no.json","w"),indent=4,ensure_ascii=False)
    json.dump(data2,open("examples/dataset/data/drop/drop_benchmark_fc_yes_share_yes_blend_no.json","w"),indent=4,ensure_ascii=False)
        
check()



