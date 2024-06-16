from detectron2.data import MetadataCatalog

# 获取所有已注册的数据集名称
dataset_names = MetadataCatalog.list()
print(dataset_names)
