import json
import matplotlib.pyplot as plt
import numpy as np



def clean(data_path,save_path,sim_threshold=0.75,token_reused_threshold=50):
    
    data = json.load(open(data_path))
    all_texts =  data["all_texts"]
    similar_pairs = data["similar_pairs"]
    
    save_data = []
    # 收集数据用于分析
    similarity_values = []
    token_reused_values = []
    sample_densities = []
    
    for pair in similar_pairs:
        target_text = all_texts[str(pair["id"])]
        if len(pair["high_similarity_top5"]) == 0 or len(pair["high_token_reused_top5"]) == 0:
            continue
        sim_top1 = pair["high_similarity_top5"][0]
        reused_top1 = pair["high_token_reused_top5"][0]
        
        if sim_top1["similarity"] < sim_threshold:
            continue
        if reused_top1["token_reused"] < token_reused_threshold:
            continue
        
        sim_top1["text"] = all_texts[str(sim_top1["id"])]
        reused_top1["text"] = all_texts[str(reused_top1["id"])]
        if sim_top1["id"] == reused_top1["id"]:
            continue
        
        # 计算样本密度（这里用文本长度作为参考）
        density = len(target_text.split())
        
        similarity_values.append(sim_top1["similarity"])
        token_reused_values.append(reused_top1["token_reused"])
        sample_densities.append(density)
        
        save_data.append({
            "target_text": {
                "id":pair["id"],
                "text":target_text
                },
            "sim_top1": sim_top1,
            "reused_top1":reused_top1
        })
        
    json.dump(save_data,open(save_path,"w"),indent=4,ensure_ascii=False)
    
    # 绘制关系图
    plt.figure(figsize=(12, 5))
    
    # 样本密度与token_reused的关系
    plt.subplot(1, 2, 1)
    plt.scatter(sample_densities, token_reused_values, alpha=0.5)
    plt.xlabel('样本密度（词数）')
    plt.ylabel('Token重用数量')
    plt.title('样本密度与Token重用关系')
    
    # 样本密度与相似度的关系
    plt.subplot(1, 2, 2)
    plt.scatter(sample_densities, similarity_values, alpha=0.5)
    plt.xlabel('样本密度（词数）')
    plt.ylabel('相似度')
    plt.title('样本密度与相似度关系')
    
    plt.tight_layout()
    plt.savefig('examples/pipeline/images/analysis_plot.png')
    
    # 计算相关系数
    density_reused_corr = np.corrcoef(sample_densities, token_reused_values)[0,1]
    density_sim_corr = np.corrcoef(sample_densities, similarity_values)[0,1]
    
    print(f"样本密度与token重用相关系数: {density_reused_corr:.3f}")
    print(f"样本密度与相似度相关系数: {density_sim_corr:.3f}")

    # Calculate reverse cumulative distribution
    token_values = np.linspace(0, max(token_reused_values), 1000)
    similarity_thresholds = np.linspace(0, max(similarity_values), 1000)
    token_density_values = []
    sim_density_values = []
    
    # Calculate both distributions
    for token_threshold in token_values:
        density = (np.array(token_reused_values) >= token_threshold).mean() * 100
        token_density_values.append(density)
    
    for sim_threshold in similarity_thresholds:
        density = (np.array(similarity_values) >= sim_threshold).mean() * 100
        sim_density_values.append(density)
    
    # Create figure with two Y axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    
    # Plot token reuse distribution on first Y axis
    line1 = ax1.plot(token_density_values, token_values, 'b-', label='Token Reuse')
    ax1.set_xlabel('Sample Density (%)')
    ax1.set_ylabel('Token Reuse Count', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Plot similarity distribution on second Y axis
    line2 = ax2.plot(sim_density_values, similarity_thresholds, 'r-', label='Similarity')
    ax2.set_ylabel('Similarity Score', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    plt.grid(True)
    plt.title('Distribution of Token Reuse and Similarity')
    
    # Add legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    # Add key density points annotations
    key_densities = [1, 5, 10, 25, 50]
    
    # Annotate token reuse points
    for density in key_densities:
        idx = np.argmin(np.abs(np.array(token_density_values) - density))
        token_value = token_values[idx]
        ax1.plot([density], [token_value], 'bo')
        ax1.annotate(f'{density}%: {token_value:.0f}', 
                    xy=(density, token_value), 
                    xytext=(10, 10),
                    textcoords='offset points',
                    color='blue')
    
    # Annotate similarity points
    for density in key_densities:
        idx = np.argmin(np.abs(np.array(sim_density_values) - density))
        sim_value = similarity_thresholds[idx]
        ax2.plot([density], [sim_value], 'ro')
        ax2.annotate(f'{density}%: {sim_value:.3f}', 
                    xy=(density, sim_value), 
                    xytext=(10, -10),
                    textcoords='offset points',
                    color='red')
    
    plt.savefig('examples/pipeline/images/distribution_comparison.png')
    
    # Print statistics
    print(f"Total samples: {len(token_reused_values)}")
    print("\nKey density point statistics:")
    for density in key_densities:
        idx_token = np.argmin(np.abs(np.array(token_density_values) - density))
        idx_sim = np.argmin(np.abs(np.array(sim_density_values) - density))
        token_value = token_values[idx_token]
        sim_value = similarity_thresholds[idx_sim]
        print(f"Density {density}%: Token Reuse = {token_value:.0f}, Similarity = {sim_value:.3f}")


def plot(save_path):
    pass

if __name__=="__main__":
    # data_path = "examples/dataset/data/insturctionv2/instruction_wildv2_similar_250331.json"
    # save_path = "examples/dataset/data/insturctionv2/instruction_wildv2_similar_250331_clean.json"
    # clean(data_path,save_path)
    
    data_path = "examples/dataset/data/sharegpt/sharegpt90k_similar_250331.json"
    save_path = "examples/dataset/data/sharegpt/sharegpt90k_similar_250331_clean.json"
    clean(data_path,save_path)
    