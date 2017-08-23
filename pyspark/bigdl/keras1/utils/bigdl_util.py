from bigdl.util.common import to_list
import uuid

def get_uid():
    return str(uuid.uuid4())

def to_bigdl(x):
    return [i.B for i in to_list(x)]