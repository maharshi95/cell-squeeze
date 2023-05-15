# %%
import struct
import numpy as np
from BitVector import BitVector

unpack_formats = {
    8: ">B",
    16: ">H",
    32: ">I",
    64: ">Q",
}

dtypes = {
    8: np.uint8,
    16: np.uint16,
    32: np.uint32,
    64: np.uint64,
}


def long_int_to_int_vector(num, bit_length=64):
    # Determine the number of 64-bit integers needed to represent the long int
    assert bit_length % 8 == 0, "bit_length must be a multiple of 8"
    word_size = bit_length // 8

    num_words = (num.bit_length() + bit_length - 1) // bit_length

    # Pack the long int into binary format using big-endian byte order
    total_bytes = num_words * word_size
    packed = num.to_bytes(total_bytes, byteorder="big")

    word_list = []
    unpack_format = unpack_formats[bit_length]
    dtype = dtypes[bit_length]

    for i in range(num_words):
        start = i * word_size
        end = start + word_size
        segment = packed[start:end]
        word = struct.unpack(unpack_format, segment)[0]
        word_list.append(word)

    return np.array(word_list, dtype=dtype)


def int_vector_to_long_int(vector, bit_length=64):
    assert bit_length % 8 == 0, "bit_length must be a multiple of 8"
    segments = []
    format = unpack_formats[bit_length]
    for word in vector:
        segments.append(struct.pack(format, word))
    packed = b"".join(segments)
    return int.from_bytes(packed, byteorder="big")


class IntVector:
    def __init__(self, num_elements: int, bit_width: int):
        self.bit_width = bit_width
        self.bitvec = BitVector(size=num_elements * bit_width)

    def __getitem__(self, index: int):
        if type(index) == int:
            start = index * self.bit_width
            end = start + self.bit_width
            return self.bitvec[start:end].intValue()

        if type(index) == slice:
            start, end = index.start, index.stop
            new_len = len(self)
            new_start, new_end = None, None
            if start is not None:
                new_start = start * self.bit_width
                new_len -= start
            if end is not None:
                new_end = end * self.bit_width
                new_len -= len(self) - end
            new_vec = IntVector(new_len, self.bit_width)
            new_vec.bitvec = self.bitvec[slice(new_start, new_end)]
            return new_vec

    def __setitem__(self, index: int, value: int):
        start = index * self.bit_width
        end = start + self.bit_width
        self.bitvec[start:end] = BitVector(intVal=value, size=self.bit_width)

    def __len__(self):
        return self.bitvec.length() // self.bit_width

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def to_numpy(self):
        return np.array(list(self), dtype=np.uint32)

    def bitword_array(self, word_length: int = 32):
        """Return a numpy array of bit words of size word_size.

        Args:
            word_length: The number of bits in each word.
        """
        N = self.bitvec.length()
        num_words = (N + word_length - 1) // word_length
        word_size = word_length // 8

        unpack_format = unpack_formats[word_length]
        dtype = dtypes[word_length]

        word_list = [None] * num_words
        for j, end in enumerate(range(N, 0, -word_length)):
            start = max(end - word_length, 0)
            int_chunk = self.bitvec[start:end].intValue()
            segment = int_chunk.to_bytes(word_size, "big")
            word = struct.unpack(unpack_format, segment)[0]
            word_list[-j - 1] = word
            # print(start, end, int_chunk, segment, word)
        return np.array(word_list, dtype=dtype)

    @classmethod
    def from_bitword_array(cls, bitword_array: np.ndarray, bit_width: int):
        word_length = np.dtype(bitword_array.dtype).itemsize * 8
        n_elem = bitword_array.size * word_length // bit_width
        print("bitword_array", bitword_array.size)
        print(n_elem)
        intvec = cls(n_elem, bit_width)
        end = intvec.bitvec.length()
        for i, word in enumerate(bitword_array[::-1]):
            start = end - word_length
            print(start, end, word)
            if start < 0:
                start = 0
            intvec.bitvec[start:end] = BitVector(intVal=word, size=(end - start))
            end = start
        return intvec
    
    @classmethod
    def from_numpy(self, array: np.ndarray):
        bitwidth = np.max(array).bit_length()
        int_vec = IntVector(len(array), bitwidth)
        for i, elem in enumerate(array):
            int_vec[i] = elem
        return int_vec
        

    def __repr__(self):
        return f"IntVector({len(self)}, {self.bit_width})"

    def __str__(self):
        return str(self.to_numpy())

    def __eq__(self, other):
        return self.bitvec == other.bitvec

    def __hash__(self):
        return hash(self.bitvec)


# %%
N = 20
bitwidth = 3
b = IntVector(N, bitwidth)

np.random.seed(0)
for i in range(N):
    b[i] = np.random.randint(0, 2**3)
print(b.to_numpy())
nbits = 32
# print(b.bitword_array(nbits))
b2 = IntVector.from_bitword_array(b.bitword_array(nbits), bitwidth)
print(b[:])
print(b2[1:5])
print(b == b2[1 : N + 1])
# %%
b.bitvec.byteArray

# %%

def make_goopdict(mat: np.ndarray):
    