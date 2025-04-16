import torch

from reed import ReED
from pipeline_imagenet.model_zoo import build_model

_verbose = False

def build_teachers(model_list, image_size, num_classes):
    teachers = []
    for model_type in model_list:
        teacher = build_model(model_type.lower(), is_pretrained=True, image_size=image_size, num_classes=num_classes)
        if _verbose: print(teacher)
        teachers.append(teacher)
    return teachers

def build_student(model_type, image_size, num_classes):
    try:
        student = build_model(model_type, is_pretrained=False, image_size=image_size, num_classes=num_classes)
    except Exception as e:
        print(f"Exception: {e}; Trying to load model from customized library.")
        from backbones import Network
        student = Network(model_type.lower(), image_size=image_size, num_classes=num_classes)
    if _verbose: print(student)
    return student

def build_distillation(student, teachers, dist_config, image_size, num_classes):
    teachers = build_teachers(teachers, image_size=image_size, num_classes=num_classes)
    student = build_student(student, image_size=image_size, num_classes=num_classes)
    model = ReED(student, teachers, dist_config, dummy_input=torch.randn(1, 3, image_size, image_size))
    return model