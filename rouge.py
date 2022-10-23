import evaluate
_metric = evaluate.load('rouge')

def score(candidates, references):
    '''
    calculates rouge scores
    copied function from hugging face transformers run_summarization.py
    '''
    result = _metric.compute(predictions=candidates, references=references, use_stemmer=True)
    result.update((key, val * 100) for key, val in result.items())
    return result

if __name__ == '__main__':
    print(_metric)
