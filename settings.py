"""
paths defined here are used in many places
"""

DATA_PATH = 'datasets'

EXPORT_PATH = 'logs'

COCO_TRAIN = 'train2014'
COCO_VAL = 'val2014'

HPatches_SRC = 'v_maskedman/1.ppm'
HPatches_DST = 'v_maskedman/4.ppm'

DEFAULT_SETTING = {
    'model': {
        'name': 'UnsuperPoint_single',
        'correspondence_threshold': 4,
        'usp_loss': {
            'alpha_usp': 1,
            'alpha_position': 1,
            'alpha_score': 2
        },
  
        'unixy_loss': {
            'alpha_unixy' : 100
        },
  
        'desc_loss': {
            'alpha_desc' : 0.001,
            'lambda_d' : 250,
            'margin_positive' : 1,
            'margin_negative' : 0.2
        },
    
        'decorr_loss':{
            'alpha_decorr' : 0.03
        }
    }
}