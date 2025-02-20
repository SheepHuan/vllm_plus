import json
from datasets import load_dataset
import tqdm
import random

dataset = load_dataset("wmt/wmt19", 'zh-en')

train_dataset = dataset['train']
test_dataset = dataset['validation']


save_path = 'examples/bench_cache/data/wmt19_zh_en.json'

save_data = []
for i in test_dataset:
    translation = i['translation']
    en = translation['en']
    zh = translation['zh']
    save_data.append({
        'en': en,
        'zh': zh
    })
  
save_data = sorted(save_data, key=lambda x: len(x['en']), reverse=True)[:100]

save_data2 = []
for i in range(15):
    data = random.sample(save_data, 15)
    new_en = [item['en'] for item in data]
    new_zh = [item['zh'] for item in data]
    save_data2.append({
        'en': ' '.join(new_en),
        'zh': ' '.join(new_zh)
    })

save_data3 = []
for i in range(15):
    data = random.sample(save_data, 10)
    new_en = [item['en'] for item in data]
    new_zh = [item['zh'] for item in data]
    for j in range(2):
        data2 = random.sample(save_data, 10)
        new_en2 = [item['en'] for item in data2]
        new_zh2 = [item['zh'] for item in data2]
        save_data3.append({
            'en': ' '.join(new_en)+' '+' '.join(new_en2),
            'zh': ' '.join(new_zh)+' '+' '.join(new_zh2)
        })


save_data = {
    "example1": save_data,
    "example2": save_data2,
    "example3": save_data3
}

json.dump(save_data, open(save_path, 'w'),indent=4,ensure_ascii=False)
    
# import evaluate

# predictions = ["hello there general kenobi", "foo bar foobar"]
# references = [
#     ["hello there general kenobi", "hello there !"],
#     ["foo bar foobar"]
# ]
# bleu = evaluate.load("bleu")
# results = bleu.compute(predictions=predictions, references=references)
# print(results)


# meteor = evaluate.load('meteor')
# predictions = ["It is a guide to action which ensures that the military always obeys the commands of the party"]
# references = ["It is a guide to action that ensures that the military will forever heed Party commands"]
# results = meteor.compute(predictions=predictions, references=references)
# print(results)