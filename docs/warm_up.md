## Learning Rate Warm-Up & Decay in Transformer Training

### Overview
In transformer training, the learning rate (LR) is not kept constant. Instead, it follows a warm-up → peak → decay schedule to ensure numerical stability and effective learning.

---

### Warm-Up Phase

- Initial LR starts near zero.
- LR increases linearly over a fixed number of warm-up steps.
- Purpose:
    - Stabilize attention and embedding layers
    - Prevent gradient explosions
    - Allow optimizer (Adam) statistics to initialize properly

**Formula:**
```
lr(t) = max_lr × (t / warmup_steps)
```

---

### Peak Learning Rate

- Reached at the end of warm-up.
- Marks the point where the model becomes stable enough for aggressive learning.
- Most meaningful representation learning begins here.

---

### Decay Phase

- After warm-up, LR gradually decreases until training ends.
- Common decay strategies:
    - Linear decay
    - Cosine decay (preferred for LMs)
- Purpose:
    - Large updates early → exploration
    - Small updates late → fine-grained refinement
    - Prevents overwriting learned structure

---

### Best Practices

- Warm-up steps: 1–5% of total training steps
- Warm-up is essential when:
    - Training from scratch
    - Using large batch sizes
    - Using higher learning rates
    - Training deep transformer models

---