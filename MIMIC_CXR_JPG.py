from matplotlib import pyplot as plt
import cv2
from pathlib import Path
import numpy as np
import pandas as pd
import os


class MIMIC_CXR_JPG:
    def __init__(
        self, path="./gcloud/mimic-cxr-jpg-2.0.0.physionet.org", init_now=False
    ) -> None:
        mimic_cxr_jpg = Path("./gcloud/mimic-cxr-jpg-2.0.0.physionet.org")
        self.path = path
        if init_now:
            self.init_data()

    def init_data(self):
        mimic_cxr_jpg_files = mimic_cxr_jpg / "files"
        self.data = {}
        for big_folder in mimic_cxr_jpg_files.glob("*"):
            self.data[big_folder.name] = {
                "path": str(big_folder),
            }
            for patient_id in big_folder.glob("*"):
                self.data[big_folder.name][patient_id.name] = {
                    "path": str(patient_id),
                }
                for study_id in patient_id.glob("*"):
                    self.data[big_folder.name][patient_id.name][study_id.name] = {
                        "path": str(study_id),
                        "imgs": [str(x) for x in study_id.glob("*.jpg")],
                    }

    def __getitem__(self, key):
        if type(key) is str:
            return self.data[key]
        else:
            return self.data[list(self.data.keys())[key]]

    def __len__(self):
        return len(self.data.keys())

    def get_stats_overview(
        self,
    ):
        big_folders = self.data.values()
        patients = [
            patient
            for big_folder in big_folders
            for patient in big_folder.values()
            if type(patient) is dict
        ]
        studies = [
            study
            for big_folder in big_folders
            for patient in big_folder.values()
            if type(patient) is dict
            for study in patient.values()
            if type(study) is dict
        ]
        images = [
            img
            for big_folder in big_folders
            for patient in big_folder.values()
            if type(patient) is dict
            for study in patient.values()
            if type(study) is dict
            for img in study["imgs"]
        ]
        num_big_folders = len(big_folders)
        num_patients = len(patients)
        num_studies = len(studies)
        num_images = len(images)
        number_overview = {
            "num_big_folders": num_big_folders,
            "num_patients": num_patients,
            "num_studies": num_studies,
            "num_images": num_images,
        }

        images_per_study = np.array([len(study["imgs"]) for study in studies])
        images_study_hist = list(np.bincount(images_per_study))
        number_overview["images_study_hist"] = images_study_hist
        return number_overview

    def get_extremelar_sample(self):
        studies = [
            study
            for big_folder in self.data.values()
            for patient in big_folder.values()
            if type(patient) is dict
            for study in patient.values()
            if type(study) is dict
        ]
        images_per_study = np.array([len(study["imgs"]) for study in studies])
        images_study_hist = np.bincount(images_per_study)
        exemplaryet = np.zeros(len(images_study_hist))
        exemplaryet[images_study_hist > 0] = 1
        exemplar = {k: "" for k, _ in enumerate(exemplaryet)}
        for i, study in enumerate(studies):
            l = len(study["imgs"])
            if exemplaryet[l] == 1:
                exemplar[l] = str(study["path"])
                exemplaryet[l] -= 1
            if exemplaryet.sum() == 0:
                break
        return {k: v for k, v in exemplar.items() if v != ""}

    def get_example_structure(self):
        sample = {}
        for k, v in self.data.items():
            sample[k] = {"path": v["path"]}
            for k1, v1 in v.items():
                if type(v1) is dict:
                    sample[k][k1] = {"path": v1["path"]}
                    for k2, v2 in v1.items():
                        if type(v2) is dict:
                            sample[k][k1][k2] = {"path": v2["path"], "imgs": v2["imgs"]}
                            break
                    break
            break
        return sample

    def get_row_from_image_ids(self, image_ids):
        """
        image_ids: list of image ids
        """
        meta = pd.read_csv(
            "./gcloud/mimic-cxr-jpg-2.0.0.physionet.org/mimic-cxr-2.0.0-metadata.csv"
        )
        meta = meta[meta["dicom_id"].isin(image_ids)]
        # get list of patient ids from meta as list
        study_ids = meta["study_id"].unique().tolist()

        chexpert = pd.read_csv(
            "./gcloud/mimic-cxr-jpg-2.0.0.physionet.org/mimic-cxr-2.0.0-chexpert.csv"
        )
        negbio = pd.read_csv(
            "./gcloud/mimic-cxr-jpg-2.0.0.physionet.org/mimic-cxr-2.0.0-negbio.csv"
        )
        # get chexpert and negbio rows for the patient ids and study ids
        chexpert = chexpert[chexpert["study_id"].isin(study_ids)]
        negbio = negbio[negbio["study_id"].isin(study_ids)]
        return meta, chexpert, negbio

    def pull_sample_from_dicom_id(self, dicom_id, saved_dir):
        """
        dicom_id: dicom id of the image
        saved_dir: directory where the image will be saved
        """
        saved_dir = os.path.join(saved_dir, "images")
        os.makedirs(saved_dir, exist_ok=True)
        meta = pd.read_csv(
            "./gcloud/mimic-cxr-jpg-2.0.0.physionet.org/mimic-cxr-2.0.0-metadata.csv"
        )
        meta = meta[meta["dicom_id"].isin([dicom_id])]
        image_path = os.path.join(
            self.path,
            "files",
            f'p{str(meta["subject_id"].unique()[0])[:2]}',
            "p" + str(meta["subject_id"].unique()[0]),
            "s" + str(meta["study_id"].unique()[0]),
            str(meta["dicom_id"].unique()[0]) + ".jpg",
        )
        saved_path = os.path.join(saved_dir, f"{dicom_id}.jpg")
        os.system(f"cp {image_path} {saved_path}")


if __name__ == "__main__":
    # Overall stats
    mimic_cxr_jpg = MIMIC_CXR_JPG()
    print(mimic_cxr_jpg.get_example_structure())
    # print(mimic_cxr_jpg.get_stats_overview())
    # print(mimic_cxr_jpg.get_extremelar_sample())

    # csv
    # chexpert = pd.read_csv("./gcloud/mimic-cxr-jpg-2.0.0.physionet.org/mimic-cxr-2.0.0-chexpert.csv")
    # print(chexpert.head())
    # print(chexpert.shape)
    # negbio = pd.read_csv("./gcloud/mimic-cxr-jpg-2.0.0.physionet.org/mimic-cxr-2.0.0-negbio.csv")
    # print(negbio.head())
    # print(negbio.shape)
