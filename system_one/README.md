### Jev-style System One — Labs 1 and 2
This project evolves a structured-output SLM baseline (Lab 1) into a typed decision interface (Lab 2).

---

Setup (Windows PowerShell)

Install Ollama from https://ollama.com/download, then run:

ollama pull qwen3:0.6b

python -m venv .venv

.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

pip install -r requirements.txt

Run the baseline and lab 2 engine:

python lab1_baseline.py

python system_one_local.py

pytest -v

---


### Lab 2 concepts

- Choice: pick one option from a closed, unordered set.

- Score: choose a level on an ordered rubric; criteria are listed low-to-high.

- Noul: estimate the probability that one yes/no proposition is true.

- DecisionRequest: carries one shared state and keyed, typed questions.

- asyncio.gather: evaluates independent questions concurrently; each gets the same state and its own question contract.

- Per-question JSON Schemas narrow the allowed return structure.
Important limitations

- Each question currently triggers an independent structured-generation call. The calls are concurrent, but they may still compete for local inference resources. This is not Jev's single-request/one-pass inference path.

- The Choice output contains only the selected option; the Score output contains a selected level index and label. Probability distributions and the weighted-mean score are not implemented in Lab 2.

- Noul is returned as noul_estimate with an explicit uncalibrated marker. Do not use it as a calibrated probability for consequential automatic decisions.

- This is an educational interface inspired by public System One concepts, not a reproduction of Jev's proprietary weights or training.

- Lab 3 will investigate candidate scoring/logits to reduce dependence on generated labels and derive distributions where the model/runtime exposes the required information.

### Tests
The tests use a fake async client so the contract and concurrency behavior can be checked without a running model. They do not benchmark actual SLM accuracy or latency