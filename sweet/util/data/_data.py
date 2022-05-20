from pathlib import Path
from functools import lru_cache
import re


DATA_PATH = (Path(__file__).parents[3] / 'data' / 'aberrometer').resolve()
ZERNIKE_TEMPLATE = re.compile(
    "Map(?P<map_num>.*)Zernike Graph\(OPD\)\n" +
    "(?P<_exam_line>(.*\n))" +
    "Zone : (?P<zone>[0-9\.]*)mm\n" +
    # "(?P<title_lines>(.*\n){5})" +
    "(?P<order_line>(.*\n))" +
    "VD :  (?P<VD>[0-9\.]*)mm\n" +
    "ZSph,ZCyl,ZAxis(?P<other_names>.*\n)" +
    "(?P<ZSph>[0-9\.\s-]*),(?P<ZCyl>[0-9\.\s-]*),(?P<ZAxis>[0-9\.\s-]*),(?P<other_vals>.*\n)"
    "(?P<title_line>(.*\n))" +
    '\n'.join(f"(?P<title{i}>{i},[^,]*,[^,]*,)\s*(?P<Z{i}>[0-9-\\.]*).*" for i in range(28))
)

@lru_cache()
def _load_all_zernike(
    participant_id: str,
    eye: str ='L'
) -> list:
    files = DATA_PATH.glob(f'{participant_id}/*Wavefront Summary*_{eye}_*.csv')

    zernike_vals = []
    for f in files:
        with open(f) as inp:
            txt = inp.read()
            for m in ZERNIKE_TEMPLATE.finditer(txt):
                d = m.groupdict()
                zernike_vals.append(d)

    return zernike_vals

@lru_cache()
def load_zernike(
    participant_id: str,
    eye: str ='L',
    bigger_one: bool = True,
) -> dict:
    """Load zernike coeffs by (anonymized) participant_id

    :param str participant_id: id of participant (e.g. ABCD)
    :param str eye: eye ('L' or 'R')
    :param bool bigger_one: get the bigger one file

    :return dict: coefs + D0
    """

    vals = _load_all_zernike(participant_id, eye)
    assert len(vals) == 2, f"Unexpected Zernike's count {len(vals)} for the eye {participant_id}-{eye}"
    if float(vals[0]['zone']) >= float(vals[1]['zone']):
        big, small = vals[0], vals[1]
    else:
        big, small = vals[1], vals[0]
    val = big if bigger_one else small

    zernike_info = {
        f'Z{i}': float(val[f'Z{i}'])
        for i in range(28)
    }
    zernike_info['D0'] = float(val['zone'])

    zernike_info['ZSph'] = float(val['ZSph'])
    zernike_info['ZCyl'] = float(val['ZCyl'])
    zernike_info['ZAxis'] = float(val['ZAxis'])

    assert float(val['VD']) == 12.0, 'Unexpected VD'

    return zernike_info


if __name__ == '__main__':
    # for testing, TODO: move to sweet/tests
    z = _load_all_zernike('KHRS', 'L')
    z = load_zernike('KHRS', 'L')
    print(z)
