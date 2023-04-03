# Subtest 1
Subtest 1 là: img -> transcript. 
Mình sẽ cần sửa vocab cho kèm luôn dấu chấm. 
Ví dụ từ UI CXR
```json
{"train": [{"id": "CXR2384_IM-0942", "report": "The heart size and pulmonary vascularity appear within normal limits. A large hiatal hernia is noted. The lungs are free of focal airspace disease. No pneumothorax or pleural effusion is seen. Degenerative changes are present in the spine.", "image_path": ["CXR2384_IM-0942/0.png", "CXR2384_IM-0942/1.png"], "split": "train"}, {"id": "CXR2926_IM-1328", "report": "Card
```
You should look into the subtest1 folder to read how they clean/format the vocab. We can start with that to build vocab for this subtest.

Anyway, we must create this annotation.json-liked file first, then we can convert it into any repo later.
