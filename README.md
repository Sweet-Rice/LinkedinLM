


# LinkedinLM

###TODO:
```
implement the gpt, training harness, etc
```


## a small LM that generates LinkedIn slop posts

This project is an educational exercise so I can learn LLM architecture.

The model consists of:
- deterministic, minbpe derived tokenizer with a gpt4 based regex boundary
- synthetic dataset using chatgpt
  

## The Tokenizer
The tokenizer is rudimentary. It's educational progression and BPE algorithm was inspired by Andrej Karpathy's minbpe. LinkedinLM is an independent implementation with its own module structure, deterministic tie-breaking (though simple as a lexicographically-shortest tie break), tests, corpus pilepine, and planned downstream eval. 

I plan to eval and run benchmarks against both versions of minbpe and tiktoken, and in the future I would like to continue developing this further for fun, though I may pivot to a more serious LM.


future developments:
Optimization of the tokenizer training


