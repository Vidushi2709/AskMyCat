
# **AskMyCat**  =^.^=  

> "Because even cats know when not to guess"  =^o^=

**AskMyCat** is a healthcare RAG that **only answers when the evidence is strong**.
If it’s unsure… it just stares at you like a judgmental cat.  `>_<`

---

## **What it does**  o_o

AskMyCat retrieves medical knowledge and checks its **EBM score** before speaking.

* `[YES]` Strong evidence → Answer incoming!  `^_^v`
* `[NO]` Weak evidence → “Meh… I dunno”  `-_-`

Perfect for:

* Sleep-deprived medical students  `@_@`
* Researchers who hate hallucinations  `O_o`
* Doctors who like chaos but also safety  `>o<`

---

## **How it works**  0_o

```
User Question  :3
    |
    v
ChromaDB Retriever  -.- 
    |
    v
EBM Scorer  o_O
    |
    v
If evidence is strong  ^_^
      -> LLM answers  =^.^=
Else  >_<  
      -> "Meh… not enough evidence"  -_-
```

---

## **Core Components**  *_*

* **Medical Dataset**  `^_~`

  * QA + explanations + references
* **ChromaDB**  `-.-`

  * Stores embeddings for fast retrieval
* **EBM Scorer**  `o_o`

  * Checks if evidence is worthy
* **LLM**  `=^.^=`

  * Only speaks if cat approves

---

## **Example Behavior**  `>_<`

**User:** “What happens to the kidney in chronic urethral obstruction?”

**AskMyCat:**

* Checks references  `o_o`
* Scores evidence  `^_^`
* High score → “The kidney shrinks because urine can’t escape… =^.^=”
* Low score → “Meh… not enough evidence -_-"

---

## **Why AskMyCat exists**  `:3`

Most medical chatbots: **“I am confident. Trust me.”**  `>.<`

AskMyCat:

> “If I don’t know, I nap.  =^_^=”

---

## **Tagline**  `=^.^=`

**“AskMyCat — it only speaks when it knows… otherwise it sleeps.”**  `-.- zzz`

---
