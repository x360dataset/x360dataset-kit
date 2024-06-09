
import os, sqlite3
from configs import get_env
cfg = get_env()   # cfg from file

class db_utils():
    def __init__(self, check_data):
        dataroot = cfg.DATA_ROOT
        reporoot = cfg.REPO_ROOT

        self.database_root = f"{dataroot}/x360dataset/x360dataset.db"
        self.original_datafolder = os.path.join(dataroot, "x360dataset")
        self.trim_datafolder = os.path.join(dataroot, "Extractx360dataset")

        utils_root = f"{reporoot}/db_utils"

        self.cache_root = os.path.join(utils_root, "cache.npy")
        self.cls_mapping_npy = os.path.join(utils_root, "cls_mapping.npy")
        self.temporal_mapping_npy = os.path.join(utils_root, "temporal_mapping.npy")
        self.temporal_label_json = os.path.join(utils_root, "temporal_label.json")

        self.error_reports = os.path.join(utils_root, "error_reports.txt")
        self.ids_npy = os.path.join(utils_root, "ids.npy")

        self.action_anno_root = f"{dataroot}/x360dataset_annotation"

        self.table_name = "x360"
        self.outdoor = cfg.outdoor_classes
        self.indoor  = cfg.indoor_classes

        self.db = sqlite3.connect(self.database_root)
        self.set_key_from_db()
        self.get_idlist()

        self.cache_cut_datas = {}
        self.train_ids = []
        self.val_ids = []
        self.test_ids = []

        self.train_id_cutwise = []
        self.val_id_cutwise = []
        self.test_id_cutwise = []

        self.temporal_label_dict = cfg.tag
        
        self.cls_mapping = {"counter": 0}
        self.temporal_mapping = {"counter": 1}

        if check_data:
            self.error_file = open(self.error_reports, "w")   # Check and filter invalid data sample
            self.check_data_qualify()
        else:
            self.load_cache()

        self.temporal_mapping[0] = "empty_cls"


    def get_idlist(self):
        cursor = self.db.cursor()
        cursor.execute("SELECT id FROM {}".format(self.table_name))
        self.idlist = cursor.fetchall()
        self.idlist = [each[0] for each in self.idlist]
        return self.idlist
    
    