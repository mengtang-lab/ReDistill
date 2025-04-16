import torch

from reed import ReED
from pipeline_imagenet.model_zoo import build_model

_verbose = False

def build_teachers(config):
    teachers = []
    for model_type in config.MODEL.TEACHERS:
        teacher = build_model(model_type.lower(), is_pretrained=True, image_size=config.DATA.IMG_SIZE, num_classes=config.MODEL.NUM_CLASSES)
        if _verbose: print(teacher)
        teachers.append(teacher)
    return teachers

def build_student(config):
    model_type = config.MODEL.STUDENT
    try:
        student = build_model(model_type, is_pretrained=False, image_size=config.DATA.IMG_SIZE, num_classes=config.MODEL.NUM_CLASSES)
    except Exception as e:
        print(f"Exception: {e}; Trying to load model from customized library.")
        from backbones import Network
        student = Network(model_type.lower(), image_size=config.DATA.IMG_SIZE, num_classes=config.MODEL.NUM_CLASSES)
    if _verbose: print(student)
    return student

def build_distillation(config):
    teachers = build_teachers(config)
    student = build_student(config)
    model = ReED(student, teachers, config.REED.DIST_CONFIG, dummy_input=torch.randn(1, 3, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE))
    return model