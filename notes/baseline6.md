# Baseline 6
Sử dụng feature extraction và attention idea từ R2gen. 

Baseline 6 sẽ đi concate với patch, thay vì 1x2048 như baseline5. 
Nhưng thay vì concatenate thành hidden*2, rồi MLP về hidden, thì mình cần concate theo chiều temporal. Tức nó là bs, patch + fix_max, hidden. Và mình sẽ cần mask đi kèm. Nghe rất nhiều work, t thấy value này nên làm sau subtest1. Do mình cần sửa:
- fuser
- Làm sao lấy feature ra length text để predict parallel cho đúng. 
Khá scary nếu m kêu t làm xong trong ... 1 tiếng .-. 

Cái R2gen cũng không nói nó đã train trên bao nhiêu IMG của mimic cxr. 