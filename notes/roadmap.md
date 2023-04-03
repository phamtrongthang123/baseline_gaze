# Roadmap 
T nghĩ nếu mình đánh vào gaze, thì nên đánh vào feature encoding. 
Flow sẽ là đi từ report generation. Contrib sẽ là feature encoding/fusing. Chứ decoding mình sẽ dùng cái đang có, và không quan tâm gì liên quan decoding nữa. 
Goal là img + gaze => feature sequence như là video. temporal_len x hidden_dim là goal. 
Có 2 loại sota mình cần coi chừng, là feature encoding cho gaze và report generation. 
Ví dụ:
- Cần coi chừng ý tưởng đi mask dần: https://aclanthology.org/2020.emnlp-main.377.pdf 
- Và ý tưởng của report generation. 

Report generation method có thể được benchmark với subtest 1 
Cái gaze encoding method kia thì hơi thốn.

Về benchmark, cần đi theo subtest 1 vì mình có thể copy số từ nhiều paper trước. 

Nói chung giờ mình đánh theo 2 goals:
- Có table số cho subtest 1 
- Add gaze encoding mà khiến score nó >= score sota, \forall sota.
