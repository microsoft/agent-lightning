import binascii


def encode_int(num, bits):
    length = ((bits + 7) // 8) * 2
    padded_hex = f"{num:0{length}x}"
    big_endian = binascii.a2b_hex(padded_hex.encode("ascii"))
    return big_endian


def decode_int(b):
    return int(binascii.b2a_hex(b), 16)
