import struct


TFLITE_IDENTIFIER = b"TFL3"
TENSOR_TYPE_FLOAT32 = 0
BUILTIN_FULLY_CONNECTED = 9


class FBWriter:
    def __init__(self):
        self.buf = bytearray(b"\x00" * 8)

    def align(self, alignment=4):
        while len(self.buf) % alignment:
            self.buf.append(0)

    def patch_uoffset(self, pos, target):
        if target <= pos:
            raise ValueError(f"uoffset must point forward: pos={pos}, target={target}")
        struct.pack_into("<I", self.buf, pos, target - pos)

    def patch_root_offset(self, table_start):
        struct.pack_into("<I", self.buf, 0, table_start)
        self.buf[4:8] = TFLITE_IDENTIFIER

    def create_string(self, text):
        data = text.encode("utf-8")
        self.align(4)
        start = len(self.buf)
        self.buf += struct.pack("<I", len(data))
        self.buf += data
        self.buf += b"\x00"
        return start

    def create_int_vector(self, values):
        self.align(4)
        start = len(self.buf)
        self.buf += struct.pack("<I", len(values))
        for value in values:
            self.buf += struct.pack("<i", int(value))
        return start

    def create_uoffset_vector(self, count):
        self.align(4)
        start = len(self.buf)
        self.buf += struct.pack("<I", count)
        positions = []
        for _ in range(count):
            positions.append(len(self.buf))
            self.buf += b"\x00\x00\x00\x00"
        return start, positions

    def start_table(self, num_fields):
        self.align(4)
        vtable_start = len(self.buf)
        vtable_size = 4 + (2 * num_fields)
        object_size = 4 + (4 * num_fields)

        self.buf += struct.pack("<HH", vtable_size, object_size)
        for _ in range(num_fields):
            self.buf += b"\x00\x00"

        table_start = len(self.buf)
        self.buf += struct.pack("<i", table_start - vtable_start)
        self.buf += b"\x00" * (object_size - 4)
        return TableBuilder(self, table_start, vtable_start)


class TableBuilder:
    def __init__(self, writer, table_start, vtable_start):
        self.writer = writer
        self.table_start = table_start
        self.vtable_start = vtable_start

    def _mark_field(self, field_index):
        field_offset = 4 + (4 * field_index)
        struct.pack_into("<H", self.writer.buf, self.vtable_start + 4 + (field_index * 2), field_offset)
        return self.table_start + field_offset

    def set_uint32(self, field_index, value):
        pos = self._mark_field(field_index)
        struct.pack_into("<I", self.writer.buf, pos, value)

    def set_int32(self, field_index, value):
        pos = self._mark_field(field_index)
        struct.pack_into("<i", self.writer.buf, pos, value)

    def set_uint8(self, field_index, value):
        pos = self._mark_field(field_index)
        self.writer.buf[pos] = value & 0xFF

    def set_uoffset(self, field_index, target):
        pos = self._mark_field(field_index)
        self.writer.patch_uoffset(pos, target)

    def reserve_uoffset(self, field_index):
        return self._mark_field(field_index)


def add_tensor(writer, vector_slot_pos, shape, name, buffer_index=0, tensor_type=TENSOR_TYPE_FLOAT32):
    tensor = writer.start_table(8)
    tensor.set_uint8(1, tensor_type)
    tensor.set_uint32(2, buffer_index)
    writer.patch_uoffset(vector_slot_pos, tensor.table_start)

    shape_vec = writer.create_int_vector(shape)
    tensor.set_uoffset(0, shape_vec)

    name_str = writer.create_string(name)
    tensor.set_uoffset(3, name_str)


def add_operator(writer, vector_slot_pos, opcode_index, inputs, outputs):
    operator = writer.start_table(3)
    operator.set_uint32(0, opcode_index)
    writer.patch_uoffset(vector_slot_pos, operator.table_start)

    inputs_vec = writer.create_int_vector(inputs)
    operator.set_uoffset(1, inputs_vec)

    outputs_vec = writer.create_int_vector(outputs)
    operator.set_uoffset(2, outputs_vec)


def build_fc3_model():
    writer = FBWriter()

    model = writer.start_table(5)
    model.set_uint32(0, 3)

    operator_codes_vec, operator_code_slots = writer.create_uoffset_vector(1)
    model.set_uoffset(1, operator_codes_vec)

    subgraphs_vec, subgraph_slots = writer.create_uoffset_vector(1)
    model.set_uoffset(2, subgraphs_vec)

    description = writer.create_string("Synthetic FC x3 test model")
    model.set_uoffset(3, description)

    buffers_vec, buffer_slots = writer.create_uoffset_vector(1)
    model.set_uoffset(4, buffers_vec)

    operator_code = writer.start_table(4)
    writer.patch_uoffset(operator_code_slots[0], operator_code.table_start)
    operator_code.set_uint8(0, BUILTIN_FULLY_CONNECTED)
    operator_code.set_int32(2, 1)
    operator_code.set_int32(3, BUILTIN_FULLY_CONNECTED)

    subgraph = writer.start_table(5)
    writer.patch_uoffset(subgraph_slots[0], subgraph.table_start)

    tensors_vec, tensor_slots = writer.create_uoffset_vector(10)
    subgraph.set_uoffset(0, tensors_vec)
    add_tensor(writer, tensor_slots[0], [1, 4], "input")
    add_tensor(writer, tensor_slots[1], [3, 4], "fc1_weight")
    add_tensor(writer, tensor_slots[2], [3], "fc1_bias")
    add_tensor(writer, tensor_slots[3], [1, 3], "fc1_output")
    add_tensor(writer, tensor_slots[4], [2, 3], "fc2_weight")
    add_tensor(writer, tensor_slots[5], [2], "fc2_bias")
    add_tensor(writer, tensor_slots[6], [1, 2], "fc2_output")
    add_tensor(writer, tensor_slots[7], [1, 2], "fc3_weight")
    add_tensor(writer, tensor_slots[8], [1], "fc3_bias")
    add_tensor(writer, tensor_slots[9], [1, 1], "fc3_output")

    inputs_vec = writer.create_int_vector([0])
    subgraph.set_uoffset(1, inputs_vec)

    outputs_vec = writer.create_int_vector([9])
    subgraph.set_uoffset(2, outputs_vec)

    operators_vec, operator_slots = writer.create_uoffset_vector(3)
    subgraph.set_uoffset(3, operators_vec)
    add_operator(writer, operator_slots[0], 0, [0, 1, 2], [3])
    add_operator(writer, operator_slots[1], 0, [3, 4, 5], [6])
    add_operator(writer, operator_slots[2], 0, [6, 7, 8], [9])

    subgraph_name = writer.create_string("main")
    subgraph.set_uoffset(4, subgraph_name)

    buffer_table = writer.start_table(1)
    writer.patch_uoffset(buffer_slots[0], buffer_table.table_start)

    writer.patch_root_offset(model.table_start)
    return bytes(writer.buf)


def main():
    output_path = "fc3_test_model.tflite"
    with open(output_path, "wb") as f:
        f.write(build_fc3_model())
    print(output_path)


if __name__ == "__main__":
    main()
