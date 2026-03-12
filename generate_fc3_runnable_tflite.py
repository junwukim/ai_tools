import struct


TFLITE_IDENTIFIER = b"TFL3"
TENSOR_TYPE_FLOAT32 = 0
BUILTIN_FULLY_CONNECTED = 9
BUILTIN_OPTIONS_FULLY_CONNECTED = 8


def align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


class TableWriter:
    def __init__(self, builder, table_start, vtable_start, field_offsets):
        self.builder = builder
        self.table_start = table_start
        self.vtable_start = vtable_start
        self.field_offsets = field_offsets

    def _mark_field(self, field_index):
        field_offset = self.field_offsets[field_index]
        struct.pack_into("<H", self.builder.buf, self.vtable_start + 4 + (field_index * 2), field_offset)
        return self.table_start + field_offset

    def set_uint32(self, field_index, value):
        pos = self._mark_field(field_index)
        struct.pack_into("<I", self.builder.buf, pos, value)

    def set_int32(self, field_index, value):
        pos = self._mark_field(field_index)
        struct.pack_into("<i", self.builder.buf, pos, value)

    def set_uint8(self, field_index, value):
        pos = self._mark_field(field_index)
        self.builder.buf[pos] = value & 0xFF

    def set_bool(self, field_index, value):
        self.set_uint8(field_index, 1 if value else 0)

    def set_uoffset(self, field_index, target):
        pos = self._mark_field(field_index)
        self.builder.patch_uoffset(pos, target)


class ForwardFlatBufferBuilder:
    def __init__(self):
        self.buf = bytearray(b"\x00" * 8)

    def align(self, alignment):
        while len(self.buf) % alignment:
            self.buf.append(0)

    def patch_uoffset(self, pos, target):
        if target <= pos:
            raise ValueError(f"uoffset must point forward: pos={pos}, target={target}")
        struct.pack_into("<I", self.buf, pos, target - pos)

    def finish(self, root_table_start):
        struct.pack_into("<I", self.buf, 0, root_table_start)
        self.buf[4:8] = TFLITE_IDENTIFIER
        return bytes(self.buf)

    def start_table(self, field_types):
        self.align(4)

        field_offsets = {}
        object_cursor = 4
        for idx, (size, alignment) in enumerate(field_types):
            object_cursor = align_up(object_cursor, alignment)
            field_offsets[idx] = object_cursor
            object_cursor += size
        object_size = align_up(object_cursor, 4)

        raw_vtable_size = 4 + (2 * len(field_types))
        vtable_size = align_up(raw_vtable_size, 4)

        vtable_start = len(self.buf)
        self.buf += struct.pack("<HH", vtable_size, object_size)
        for _ in field_types:
            self.buf += b"\x00\x00"
        if vtable_size > raw_vtable_size:
            self.buf += b"\x00" * (vtable_size - raw_vtable_size)

        table_start = len(self.buf)
        self.buf += struct.pack("<i", table_start - vtable_start)
        self.buf += b"\x00" * (object_size - 4)
        return TableWriter(self, table_start, vtable_start, field_offsets)

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

    def create_ubyte_vector(self, raw_bytes, force_align=4):
        self.align(max(4, force_align))
        start = len(self.buf)
        self.buf += struct.pack("<I", len(raw_bytes))
        self.buf += raw_bytes
        return start

    def create_uoffset_vector(self, count):
        self.align(4)
        start = len(self.buf)
        self.buf += struct.pack("<I", count)
        slots = []
        for _ in range(count):
            slots.append(len(self.buf))
            self.buf += b"\x00\x00\x00\x00"
        return start, slots


def floats_to_bytes(values):
    return struct.pack("<" + ("f" * len(values)), *values)


def add_tensor(builder, tensor_slot, shape, tensor_type, buffer_index, name):
    tensor = builder.start_table([
        (4, 4),  # shape:[int]
        (1, 1),  # type:TensorType (byte)
        (4, 4),  # buffer:uint
        (4, 4),  # name:string
    ])
    builder.patch_uoffset(tensor_slot, tensor.table_start)
    tensor.set_uint8(1, tensor_type)
    tensor.set_uint32(2, buffer_index)

    shape_vec = builder.create_int_vector(shape)
    tensor.set_uoffset(0, shape_vec)

    name_str = builder.create_string(name)
    tensor.set_uoffset(3, name_str)


def add_fc_operator(builder, operator_slot, opcode_index, inputs, outputs):
    operator = builder.start_table([
        (4, 4),  # opcode_index:uint
        (4, 4),  # inputs:[int]
        (4, 4),  # outputs:[int]
        (1, 1),  # builtin_options_type:BuiltinOptions
        (4, 4),  # builtin_options:table
    ])
    builder.patch_uoffset(operator_slot, operator.table_start)
    operator.set_uint32(0, opcode_index)
    operator.set_uint8(3, BUILTIN_OPTIONS_FULLY_CONNECTED)

    inputs_vec = builder.create_int_vector(inputs)
    operator.set_uoffset(1, inputs_vec)

    outputs_vec = builder.create_int_vector(outputs)
    operator.set_uoffset(2, outputs_vec)

    fc_options = builder.start_table([
        (1, 1),  # fused_activation_function
        (1, 1),  # weights_format
        (1, 1),  # keep_num_dims
        (1, 1),  # asymmetric_quantize_inputs
    ])
    operator.set_uoffset(4, fc_options.table_start)


def add_buffer(builder, buffer_slot, raw_bytes=None):
    buffer_table = builder.start_table([
        (4, 4),  # data:[ubyte]
    ])
    builder.patch_uoffset(buffer_slot, buffer_table.table_start)
    if raw_bytes is not None:
        data_vec = builder.create_ubyte_vector(raw_bytes, force_align=16)
        buffer_table.set_uoffset(0, data_vec)


def build_runnable_fc3_model():
    builder = ForwardFlatBufferBuilder()

    model = builder.start_table([
        (4, 4),  # version:uint
        (4, 4),  # operator_codes:[OperatorCode]
        (4, 4),  # subgraphs:[SubGraph]
        (4, 4),  # description:string
        (4, 4),  # buffers:[Buffer]
    ])
    model.set_uint32(0, 3)

    operator_codes_vec, operator_code_slots = builder.create_uoffset_vector(1)
    model.set_uoffset(1, operator_codes_vec)

    subgraphs_vec, subgraph_slots = builder.create_uoffset_vector(1)
    model.set_uoffset(2, subgraphs_vec)

    description = builder.create_string("Runnable FC x3 float32 test model")
    model.set_uoffset(3, description)

    buffers_vec, buffer_slots = builder.create_uoffset_vector(7)
    model.set_uoffset(4, buffers_vec)

    operator_code = builder.start_table([
        (1, 1),  # deprecated_builtin_code:byte
        (4, 4),  # custom_code:string
        (4, 4),  # version:int
        (4, 4),  # builtin_code:BuiltinOperator
    ])
    builder.patch_uoffset(operator_code_slots[0], operator_code.table_start)
    operator_code.set_uint8(0, BUILTIN_FULLY_CONNECTED)
    operator_code.set_int32(2, 1)
    operator_code.set_int32(3, BUILTIN_FULLY_CONNECTED)

    subgraph = builder.start_table([
        (4, 4),  # tensors:[Tensor]
        (4, 4),  # inputs:[int]
        (4, 4),  # outputs:[int]
        (4, 4),  # operators:[Operator]
        (4, 4),  # name:string
    ])
    builder.patch_uoffset(subgraph_slots[0], subgraph.table_start)

    tensors_vec, tensor_slots = builder.create_uoffset_vector(10)
    subgraph.set_uoffset(0, tensors_vec)

    inputs_vec = builder.create_int_vector([0])
    subgraph.set_uoffset(1, inputs_vec)

    outputs_vec = builder.create_int_vector([9])
    subgraph.set_uoffset(2, outputs_vec)

    operators_vec, operator_slots = builder.create_uoffset_vector(3)
    subgraph.set_uoffset(3, operators_vec)

    subgraph_name = builder.create_string("main")
    subgraph.set_uoffset(4, subgraph_name)

    add_tensor(builder, tensor_slots[0], [1, 4], TENSOR_TYPE_FLOAT32, 0, "input")
    add_tensor(builder, tensor_slots[1], [3, 4], TENSOR_TYPE_FLOAT32, 1, "fc1_weight")
    add_tensor(builder, tensor_slots[2], [3], TENSOR_TYPE_FLOAT32, 2, "fc1_bias")
    add_tensor(builder, tensor_slots[3], [1, 3], TENSOR_TYPE_FLOAT32, 0, "fc1_output")
    add_tensor(builder, tensor_slots[4], [2, 3], TENSOR_TYPE_FLOAT32, 3, "fc2_weight")
    add_tensor(builder, tensor_slots[5], [2], TENSOR_TYPE_FLOAT32, 4, "fc2_bias")
    add_tensor(builder, tensor_slots[6], [1, 2], TENSOR_TYPE_FLOAT32, 0, "fc2_output")
    add_tensor(builder, tensor_slots[7], [1, 2], TENSOR_TYPE_FLOAT32, 5, "fc3_weight")
    add_tensor(builder, tensor_slots[8], [1], TENSOR_TYPE_FLOAT32, 6, "fc3_bias")
    add_tensor(builder, tensor_slots[9], [1, 1], TENSOR_TYPE_FLOAT32, 0, "fc3_output")

    add_fc_operator(builder, operator_slots[0], 0, [0, 1, 2], [3])
    add_fc_operator(builder, operator_slots[1], 0, [3, 4, 5], [6])
    add_fc_operator(builder, operator_slots[2], 0, [6, 7, 8], [9])

    add_buffer(builder, buffer_slots[0], None)
    add_buffer(builder, buffer_slots[1], floats_to_bytes([
        0.50, -0.25, 0.75, 0.10,
        -0.40, 0.90, 0.30, -0.20,
        0.60, 0.15, -0.55, 0.80,
    ]))
    add_buffer(builder, buffer_slots[2], floats_to_bytes([0.10, -0.20, 0.05]))
    add_buffer(builder, buffer_slots[3], floats_to_bytes([
        0.70, -0.10, 0.20,
        -0.30, 0.40, 0.60,
    ]))
    add_buffer(builder, buffer_slots[4], floats_to_bytes([0.03, -0.07]))
    add_buffer(builder, buffer_slots[5], floats_to_bytes([1.10, -0.35]))
    add_buffer(builder, buffer_slots[6], floats_to_bytes([0.12]))

    return builder.finish(model.table_start)


def main():
    output_path = "fc3_runnable_model.tflite"
    with open(output_path, "wb") as f:
        f.write(build_runnable_fc3_model())
    print(output_path)


if __name__ == "__main__":
    main()
