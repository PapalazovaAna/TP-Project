---
license: cc-by-nc-4.0
task_categories:
- text-generation
language:
- en
pretty_name: Natural Reasoning
size_categories:
- 1M<n<10M
---
[NaturalReasoning](https://arxiv.org/abs/2502.13124) is a large-scale dataset for general reasoning tasks. It consists of high-quality challenging reasoning questions backtranslated from pretraining corpora [DCLM](https://github.com/mlfoundations/dclm) and [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath). The questions have been deduplicated and decontaminated from popular reasoning benchmarks including MATH, GPQA, MMLU-Pro, MMLU-STEM. For each question, we extract the reference final answer from the original document from the pretraining corpora if possible. We also provide a model-generated response from Llama3.3-70B-Instruct.

We release a 1.1 million subset of NaturalReasoning to the research community to foster research on training strong LLM reasoners. 

You can load the dataset as follows
```python
from datasets import load_dataset

ds = load_dataset("facebook/natural_reasoning")
```

For more information regarding data collection, please refer to our [paper](https://arxiv.org/abs/2502.13124).



## Reference Answer Statistics
In the 1.1 million subset, 18.29% of the questions do not have a reference answer, 9.71% of the questions have a single word answer, 21.58% of the questions have a short answer while 50.42% of the questions have a long reference answer.


## Scaling Curve
Training on NaturalReasoning shows better scaling effects than training on other datasets when training Llama3.1-8B-Instruct model. In particular, we measure the average performance on three benchmarks: MATH, GPQA, MMLU-Pro. 

<img src="https://cdn-uploads.huggingface.co/production/uploads/659a395421a7431643caedda/S6aO-agjRRhc0JLkohZ5z.jpeg" style="width:50%; max-width:400px;">



## Citation
If you use data from NaturalReasoning, please cite with the following BibTex entry:
```
@misc{yuan2025naturalreasoningreasoningwild28m,
      title={NaturalReasoning: Reasoning in the Wild with 2.8M Challenging Questions}, 
      author={Weizhe Yuan and Jane Yu and Song Jiang and Karthik Padthe and Yang Li and Dong Wang and Ilia Kulikov and Kyunghyun Cho and Yuandong Tian and Jason E Weston and Xian Li},
      year={2025},
      eprint={2502.13124},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.13124}, 
}
```
