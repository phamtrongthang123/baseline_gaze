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

subtest 1 -> subtest 2
subtest 1 -> subtest 3 
subtest 2, subtest 3 -> subtest 4 

Lý do order trên phải được tuân thủ vì chưa có gì sure là code chạy đúng
Ngoài ra mình còn có thể nhờ Jacob chạy bộ image processing để sinh ra góc nhìn mới. 
Và apply guided (cái vụ cắt bbox / blur) theo gaze với fixation center.

Nếu feature trên visualgpt nó ra tốt thì dùng lên gaze sample luôn. Tức mọi thứ nên dành thời gian vào visualgpt. 

Img from now on is the raw, not the resized version. Như thế sẽ giảm câu hỏi phải trả lời. 
Raw full size jpg -> visual extractor -> feature -> postprocessing 

Bây giờ mình cần:
- jpg -> transform r2 style -> resnet101 -> R = 49,2048 (lớp cuối) [done]
- jpg -> transform r2 style -> resnet101 (train with r2gen) -> R2 [done]
- jpg -> keep raw size -> segment anything -> R3 = 4096, 256 [done] 
- jpg -> transform r2 style but without resize 224 224 -> resnet101 -> RF = ??, 2048
- jpg -> transform r2 style but without resize 224 224 -> resnet101 (train with r2gen) -> RF2 = ??, 2048

Goal là tìm best feature extractor. Có thể mình sẽ cần set up proxy task. 
Mình có rảnh đâu mà đi tự chế lại, đống survey của Jacob mình theo thì mấy work gần đây có nào dùng được đâu .-. Có thì đã hay. Nói chứ cũng có vài repo nên xem qua (nên clone/test theo order star decreasing): 
- https://github.com/zzxslp/WCL (~14). this repo has cluster_annotation.json for mimic_cxr.
- https://github.com/zhjohnchan/R2GenCMN (30)
- https://github.com/batmanlab/AGXNet (~10)
- https://github.com/ivonajdenkoska/variational-xray-report-gen (~15)

Checked:
- R2 thì rõ rồi (109 stars)
- https://github.com/wang-zhanyu/MSAT (~10): I can't run this https://github.com/wang-zhanyu/MSAT/issues/3
- https://github.com/farrell236/RATCHET (37 stars), this one is tensorflow 

Trước khi chạy tiếp 4 repo, mình cần make sure là reproduce được r2 để đảm bảo mình có safe point. 
TODO: Chạy 4/7 repo trên. Lấy luôn visual feature nếu nó cho pretrain.





Nếu apply với gaze thì
- step 1:
    - jpg -> guided -> best feature extractor -> RG 
- step 2:
    - RG (no postprocessing)
    - RG -> RoiAlign 
- or: RG + fuser with attention gaze like in thang-baseline

Chỉ có thế mới đảm bảo được nó là temporal, nhưng cái này để sau.  






