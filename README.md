Tested with python3.11

## Installation

```python
"""
Make sure python3 points to python3.11 or above
"""
### 1. install virtual env
python3 -m venv env

### 2. activate virtual env 
source env/bin/activate

#### 3. install dependencies
pip install -r requirements.py

### 4. test models on the benchmark
./test_models.py

### 5. score responses
./score_responses.py

### 6. report stats about concepts, responses and scores
./stats.py

### 7. summarize scores
./summarize_scores.py

### 8. plot results 
./plots_results.py

### 9. investigate 'semantic-field-size' results
./investigate_sfs.py
```

