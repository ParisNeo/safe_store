import numpy as np
import json
def serialize_enum_to_json(enum_member):
    return enum_member.value

class NumpyEncoderDecoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {'__numpy_array__': True, 'data': obj.tolist()}
        return super(NumpyEncoderDecoder, self).default(obj)

    @staticmethod
    def as_numpy_array(dct):
        if '__numpy_array__' in dct:
            return np.array(dct['data'])
        return dct