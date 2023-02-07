from MIMIC_CXR_JPG import MIMIC_CXR_JPG
from reflacx import REFLACX
from egd import EGD
dcm_ids = ['9c452d99-a2e14f15-8458865f-92178374-a9e29b76', '53ba5a31-2d6bda0a-abb7ad77-065a0f20-b4fa6b44']
dcm_ids = ['5c61db80-672e0874-ba236081-53d2f8ad-ab8d2efb', '8afc3991-7ec1912b-b24d9633-3c4b0f57-17c0e8f4', '060cf092-fe76bdf7-19fee515-26cbef2c-5c16ba6f']
mimic_ = MIMIC_CXR_JPG()
saved_dir = "./exemplars"
rflx = REFLACX()
egg = EGD()
for dcm_id in dcm_ids:
    mimic_.pull_sample_from_dicom_id(dcm_id, saved_dir)
    rflx.pull_labels_from_dicom_id(dcm_id, saved_dir)
    egg.pull_labels_from_dicom_id(dcm_id, saved_dir)