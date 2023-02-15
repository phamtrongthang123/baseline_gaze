import os
from pathlib import Path
import json

import pandas as pd

# "dicom_id": {"img_path_jpg", "img_path_dcm", "transcript_egd", "gaze_egd", "transcript_reflacx", "gaze_reflacx"}

make_element = lambda patient_id, study_id, img_path_jpg, img_path_dcm, transcript_egd, gaze_egd, fixation_egd, reflacx_id, transcript_reflacx, gaze_reflacx, fixation_reflacx: {
    "patient_id": patient_id,
    "study_id": study_id,
    "img_path_jpg": img_path_jpg,
    "img_path_dcm": img_path_dcm,
    "transcript_egd": transcript_egd,
    "gaze_egd": gaze_egd,
    "fixation_egd": fixation_egd,
    "reflacx_id": [reflacx_id],
    "transcript_reflacx": [transcript_reflacx],
    "gaze_reflacx": [gaze_reflacx],
    "fixation_reflacx": [fixation_reflacx],
}

make_image_path = lambda patient_id, study_id, dicom_id: (
    "data_here/mimic-cxr/2.0.0/files/p{}/p{}/s{}/{}.dcm".format(
        patient_id[:2], patient_id, study_id, dicom_id
    ),
    "data_here/mimic-cxr-jpg-2.0.0/files/p{}/p{}/s{}/{}.jpg".format(
        patient_id[:2], patient_id, study_id, dicom_id
    ),
)


def check_existence(fp):
    if not os.path.isfile(fp):
        raise ValueError("File does not exist: {}".format(fp))


metadata_dict = {}

# reflacx
reflacx_maindir = Path("data_here/REFLACX/main_data")
reflacx_gaze_maindir = Path("data_here/REFLACX/gaze_data")
metadata_reflacx = [
    pd.read_csv(reflacx_maindir / "metadata_phase_{}.csv".format(i), dtype=str)
    for i in range(1, 4)
]


for i, df in enumerate(metadata_reflacx):
    for index, row in df.iterrows():
        dicom_id = row["dicom_id"]
        subject_id = row["subject_id"]
        study_id = (
            row["image"]
            .replace("physionet.org/files/mimic-cxr/2.0.0/files/", "")
            .split("/")[2][1:]
        )
        img_path_dcm, img_path_jpg = make_image_path(subject_id, study_id, dicom_id)
        # check if the image exists
        check_existence(img_path_dcm)
        check_existence(img_path_jpg)

        transcript_egd, gaze_egd = "", ""
        fixation_egd = ""
        reflacx_id = row["id"]
        transcript_reflacx = (
            f"{str(reflacx_maindir)}/{reflacx_id}/timestamps_transcription.csv"
        )
        gaze_reflacx = f"{str(reflacx_gaze_maindir)}/{reflacx_id}/gaze.csv"
        fixation_reflacx = f"{str(reflacx_maindir)}/{reflacx_id}/fixations.csv"
        try:
            check_existence(transcript_reflacx)
            check_existence(gaze_reflacx)
            check_existence(fixation_reflacx)
            if dicom_id not in metadata_dict:
                metadata_dict[dicom_id] = make_element(
                    subject_id,
                    study_id,
                    img_path_jpg,
                    img_path_dcm,
                    transcript_egd,
                    gaze_egd,
                    fixation_egd,
                    reflacx_id,
                    transcript_reflacx,
                    gaze_reflacx,
                    fixation_reflacx,
                )
            else:
                metadata_dict[dicom_id]["reflacx_id"].append(reflacx_id)
                metadata_dict[dicom_id]["transcript_reflacx"].append(transcript_reflacx)
                metadata_dict[dicom_id]["gaze_reflacx"].append(gaze_reflacx)
                metadata_dict[dicom_id]["fixation_reflacx"].append(fixation_reflacx)
        except:
            # we skip all files that are not available
            pass

# egd
egd_maindir = Path("data_here/egd-cxr/1.0.0")
metadata_egd = pd.read_csv(egd_maindir / "master_sheet.csv", dtype=str)
eye_gaze = pd.read_csv(os.path.join(str(egd_maindir), "eye_gaze.csv"))
fixation = pd.read_csv(os.path.join(str(egd_maindir), "fixations.csv"))
for index, row in metadata_egd.iterrows():
    dicom_id = row["dicom_id"]
    patient_id = row["patient_id"]
    study_id = row["study_id"]
    img_path_dcm, img_path_jpg = make_image_path(patient_id, study_id, dicom_id)
    transcript_egd = (
        f"{str(egd_maindir)}/audio_segmentation_transcripts/{dicom_id}/transcript.json"
    )
    rp = eye_gaze[eye_gaze["DICOM_ID"].isin([dicom_id])]
    gaze_egd = os.path.join(f"{str(egd_maindir)}/eye_gaze/{dicom_id}", f"gaze.csv")
    os.makedirs(f"{str(egd_maindir)}/eye_gaze/{dicom_id}", exist_ok=True)
    rp.to_csv(gaze_egd)

    rf = fixation[fixation["DICOM_ID"].isin([dicom_id])]
    fixation_egd = os.path.join(
        f"{str(egd_maindir)}/fixations/{dicom_id}", f"fixations.csv"
    )
    os.makedirs(f"{str(egd_maindir)}/fixations/{dicom_id}", exist_ok=True)
    rf.to_csv(fixation_egd)
    check_existence(img_path_dcm)
    check_existence(img_path_jpg)
    check_existence(transcript_egd)
    check_existence(gaze_egd)
    check_existence(fixation_egd)
    reflacx_id = ""
    transcript_reflacx = ""
    gaze_reflacx = ""
    fixation_reflacx = ""
    if dicom_id not in metadata_dict:
        metadata_dict[dicom_id] = make_element(
            subject_id,
            study_id,
            img_path_jpg,
            img_path_dcm,
            transcript_egd,
            gaze_egd,
            fixation_egd,
            reflacx_id,
            transcript_reflacx,
            gaze_reflacx,
            fixation_reflacx,
        )
    else:
        metadata_dict[dicom_id]["transcript_egd"] = transcript_egd
        metadata_dict[dicom_id]["gaze_egd"] = gaze_egd
        metadata_dict[dicom_id]["fixation_egd"] = fixation_egd


with open("data_here/gazetoy_metadata.json", "w") as f:
    json.dump(metadata_dict, f)
