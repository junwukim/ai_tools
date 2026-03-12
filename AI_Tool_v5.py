# -*- coding: utf-8 -*-
import sys
import csv
import os
import html
import struct
import serial
import serial.tools.list_ports
import datetime
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

DEFAULT_BUTTON_WIDTH = 110
DEFAULT_LABEL_WIDTH = 120
DEFAULT_REPEAT_INTERVAL = 3000
DEFAULT_TOUCH_INTERVAL = 3000
DEFAULT_MAX_LOG_LINES = 40000
APP_PACKAGE_PREFIX = "lupa.usr."
MAX_APP_COUNT = 4
TFLITE_FILE_IDENTIFIER = b"TFL3"

TFLITE_BUILTIN_OP_NAMES = {
    0: "ADD",
    1: "AVERAGE_POOL_2D",
    2: "CONCATENATION",
    3: "CONV_2D",
    4: "DEPTHWISE_CONV_2D",
    5: "DEPTH_TO_SPACE",
    6: "DEQUANTIZE",
    8: "FLOOR",
    9: "FULLY_CONNECTED",
    14: "LOGISTIC",
    17: "MAX_POOL_2D",
    18: "MUL",
    19: "RELU",
    21: "RELU6",
    22: "RESHAPE",
    25: "SOFTMAX",
    28: "TANH",
    32: "CUSTOM",
    34: "PAD",
    36: "GATHER",
    39: "TRANSPOSE",
    40: "MEAN",
    41: "SUB",
    42: "DIV",
    43: "SQUEEZE",
    45: "STRIDED_SLICE",
    47: "EXP",
    50: "LOG_SOFTMAX",
    53: "CAST",
    54: "PRELU",
    55: "MAXIMUM",
    56: "ARG_MAX",
    57: "MINIMUM",
    58: "LESS",
    59: "NEG",
    60: "PADV2",
    61: "GREATER",
    62: "GREATER_EQUAL",
    63: "LESS_EQUAL",
    64: "SELECT",
    65: "SLICE",
    67: "TRANSPOSE_CONV",
    69: "TILE",
    70: "EXPAND_DIMS",
    71: "EQUAL",
    72: "NOT_EQUAL",
    73: "LOG",
    74: "SUM",
    75: "SQRT",
    76: "RSQRT",
    77: "SHAPE",
    78: "POW",
    79: "ARG_MIN",
    82: "REDUCE_MAX",
    83: "PACK",
    88: "UNPACK",
    89: "REDUCE_MIN",
    92: "SQUARE",
    97: "RESIZE_NEAREST_NEIGHBOR",
    98: "LEAKY_RELU",
    101: "ABS",
    102: "SPLIT_V",
    103: "UNIQUE",
    104: "CEIL",
    106: "ADD_N",
    107: "GATHER_ND",
    108: "COS",
    109: "WHERE",
    114: "QUANTIZE",
    117: "HARD_SWISH",
    126: "BATCH_MATMUL",
}

TFLITE_TENSOR_TYPE_NAMES = {
    0: "FLOAT32",
    1: "FLOAT16",
    2: "INT32",
    3: "UINT8",
    4: "INT64",
    5: "STRING",
    6: "BOOL",
    7: "INT16",
    8: "COMPLEX64",
    9: "INT8",
    10: "FLOAT64",
    11: "COMPLEX128",
    12: "UINT64",
    13: "RESOURCE",
    14: "VARIANT",
    15: "UINT32",
    16: "UINT16",
    17: "INT4",
}

class RelayWorker(QThread):
    error_occurred = pyqtSignal(str)

    def __init__(self, lcp_port, osp_port):
        super().__init__()
        self.lcp_port = lcp_port
        self.osp_port = osp_port
        self.running = True

    def run(self):
        try:
            lcp = serial.Serial(self.lcp_port, 115200, timeout=1)
            osp = serial.Serial(self.osp_port, 115200, timeout=1)
            while self.running:
                if lcp.in_waiting:
                    data = lcp.read(lcp.in_waiting)
                    osp.write(data)
                if osp.in_waiting:
                    data = osp.read(osp.in_waiting)
                    lcp.write(data)
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            try:
                lcp.close()
            except: pass
            try:
                osp.close()
            except: pass

    def stop(self):
        self.running = False
        self.wait()

class SerialLogReader(QThread):
    log_received = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, port, baudrate=115200):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.running = True
        self.serial_conn = None

    def run(self):
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            while self.running:
                if self.serial_conn.in_waiting:
                    data = self.serial_conn.read(self.serial_conn.in_waiting)
                    try:
                        try:
                            text = data.decode("utf-8")
                        except UnicodeDecodeError:
                            try:
                                text = data.decode("cp949")
                            except UnicodeDecodeError:
                                text = data.decode("latin1", errors="replace")
                    except Exception:
                        text = str(data)
                    self.log_received.emit(text)
                self.msleep(50)
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()

    def stop(self):
        self.running = False
        self.wait()

class HistoryLineEdit(QLineEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = []
        self.history_index = -1

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Up:
            if self.history:
                if self.history_index == -1:
                    self.history_index = len(self.history) - 1
                elif self.history_index > 0:
                    self.history_index -= 1
                self.setText(self.history[self.history_index])
            return
        elif event.key() == Qt.Key_Down:
            if self.history:
                if self.history_index < len(self.history) - 1:
                    self.history_index += 1
                    self.setText(self.history[self.history_index])
                else:
                    self.history_index = -1
                    self.clear()
            return
        elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            text = self.text().strip()
            if text and (not self.history or self.history[-1] != text):
                self.history.append(text)
            self.history_index = -1
        super().keyPressEvent(event)

class FlatBufferTable:
    def __init__(self, data, table_pos):
        self.data = data
        self.table_pos = table_pos
        self.vtable_pos = table_pos - struct.unpack_from('<i', data, table_pos)[0]

    def has_field(self, field_index):
        return self._field_pos(field_index) is not None

    def _field_pos(self, field_index):
        vtable_length = struct.unpack_from('<H', self.data, self.vtable_pos)[0]
        slot = self.vtable_pos + 4 + (field_index * 2)
        if slot + 2 > self.vtable_pos + vtable_length:
            return None
        field_offset = struct.unpack_from('<H', self.data, slot)[0]
        if field_offset == 0:
            return None
        return self.table_pos + field_offset

    def _indirect(self, offset_pos):
        return offset_pos + struct.unpack_from('<I', self.data, offset_pos)[0]

    def get_uint32(self, field_index, default=0):
        field_pos = self._field_pos(field_index)
        if field_pos is None:
            return default
        return struct.unpack_from('<I', self.data, field_pos)[0]

    def get_int32(self, field_index, default=0):
        field_pos = self._field_pos(field_index)
        if field_pos is None:
            return default
        return struct.unpack_from('<i', self.data, field_pos)[0]

    def get_uint8(self, field_index, default=0):
        field_pos = self._field_pos(field_index)
        if field_pos is None:
            return default
        return self.data[field_pos]

    def get_string(self, field_index, default=''):
        field_pos = self._field_pos(field_index)
        if field_pos is None:
            return default
        string_pos = self._indirect(field_pos)
        string_len = struct.unpack_from('<I', self.data, string_pos)[0]
        string_bytes = self.data[string_pos + 4:string_pos + 4 + string_len]
        return string_bytes.decode('utf-8', errors='replace')

    def get_int_vector(self, field_index):
        field_pos = self._field_pos(field_index)
        if field_pos is None:
            return []
        vector_pos = self._indirect(field_pos)
        vector_len = struct.unpack_from('<I', self.data, vector_pos)[0]
        if vector_len == 0:
            return []
        return list(struct.unpack_from(f'<{vector_len}i', self.data, vector_pos + 4))

    def get_table_vector(self, field_index):
        field_pos = self._field_pos(field_index)
        if field_pos is None:
            return []
        vector_pos = self._indirect(field_pos)
        vector_len = struct.unpack_from('<I', self.data, vector_pos)[0]
        items = []
        item_pos = vector_pos + 4
        for idx in range(vector_len):
            offset_pos = item_pos + (idx * 4)
            items.append(FlatBufferTable(self.data, self._indirect(offset_pos)))
        return items

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.max_apps = MAX_APP_COUNT
        self.app_names = [None] * self.max_apps
        self.app_start_bts = [None] * self.max_apps
        self.app_quit_bts = [None] * self.max_apps
        self.start_repeat_bts = [None] * self.max_apps
        self.is_app_running = [False] * self.max_apps
        self.repeat_flags = [False] * self.max_apps
        self.repeat_timers = [QBasicTimer() for _ in range(self.max_apps)]
        self.repeat_counts = [0] * self.max_apps
        self.last_meminfo_value = 0
        self.touch_timer = QBasicTimer()
        self.log_reader = None
        self.relay_worker = None
        self.mem_check = False
        self.auto_touch_running = False
        self.section_widgets = {}
        self.section_checkboxes = {}
        self.visibility_checkbox_order = []
        self._buttons_enabled = False
        self.app_active_count = 1
        self.initUI()

    def make_label(self, text, width=DEFAULT_LABEL_WIDTH, bgcolor="#e0e0e0", color="black", bold=False):
        label = QLabel(f"<b>{text}</b>" if bold else text)
        label.setFixedWidth(width)
        label.setStyleSheet(f"color:{color}; background:{bgcolor};")
        label.setAlignment(Qt.AlignCenter)
        return label

    def make_button(self, text, width=DEFAULT_BUTTON_WIDTH, bgcolor="#e0e0e0", color="black", handler=None):
        btn = QPushButton(text)
        btn.setFixedWidth(width)
        btn.setStyleSheet(f"background-color:{bgcolor}; color:{color};")
        if handler:
            btn.clicked.connect(handler)
        return btn

    def get_full_app_name(self, raw_name):
        name = raw_name.strip()
        if not name:
            return ''
        if name.startswith(APP_PACKAGE_PREFIX):
            return name
        return f"{APP_PACKAGE_PREFIX}{name}"

    def get_app_suffix(self, full_name):
        if not full_name:
            return ''
        if full_name.startswith(APP_PACKAGE_PREFIX):
            return full_name[len(APP_PACKAGE_PREFIX):]
        return full_name

    def make_section_widget(self, layout):
        widget = QWidget()
        if hasattr(layout, "setContentsMargins"):
            layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(layout)
        return widget

    def create_visibility_control_widget(self, section_definitions):
        panel = QWidget()
        root_layout = QVBoxLayout(panel)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(4)

        header_layout = QHBoxLayout()
        header_layout.addWidget(self.make_label('View', width=60))
        self.all_toggle_btn = self.make_button('All select', width=110, handler=self.toggle_all_sections)
        header_layout.addWidget(self.all_toggle_btn)
        header_layout.addStretch()
        root_layout.addLayout(header_layout)

        self.section_checkboxes = {}
        self.section_widgets = {}
        self.washer_control_section_keys = ('relay', 'washer')
        washer_control_hidden_keys = set(self.washer_control_section_keys)
        self.washer_control_cb = QCheckBox('Washer Test')
        self.washer_control_cb.setChecked(False)
        self.washer_control_cb.toggled.connect(self.toggle_washer_control_sections)

        grid_layout = QGridLayout()
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setHorizontalSpacing(12)
        grid_layout.setVerticalSpacing(4)

        section_name_map = {}
        for key, name, widget in section_definitions:
            cb = QCheckBox(name)
            cb.setChecked(False)
            cb.toggled.connect(lambda checked, k=key: self.toggle_section_visibility(k, checked))
            self.section_checkboxes[key] = cb
            self.section_widgets[key] = widget
            widget.setVisible(False)
            section_name_map[key] = cb
            if key in washer_control_hidden_keys:
                cb.setVisible(False)

        ordered_keys = []
        requested_order = getattr(self, 'visibility_checkbox_order', [])
        if requested_order:
            for key in requested_order:
                if key == 'washer_test':
                    ordered_keys.append('washer_test')
                elif key in section_name_map and key not in washer_control_hidden_keys:
                    ordered_keys.append(key)
            for key in section_name_map:
                if key in washer_control_hidden_keys or key in ordered_keys:
                    continue
                ordered_keys.append(key)
        else:
            ordered_keys = [
                key for key, _ in sorted(
                    [(key, section_name_map[key].text().lower()) for key in section_name_map.keys() if key not in washer_control_hidden_keys]
                    + [('washer_test', 'washer test')],
                    key=lambda item: item[1]
                )
            ]
        if 'washer_test' not in ordered_keys:
            ordered_keys.append('washer_test')

        # remove duplicates while preserving order
        deduped_keys = []
        seen = set()
        for key in ordered_keys:
            if key in seen:
                continue
            deduped_keys.append(key)
            seen.add(key)
        ordered_keys = deduped_keys

        visible_idx = 0
        for key in ordered_keys:
            if key == 'washer_test':
                cb = self.washer_control_cb
            else:
                cb = section_name_map.get(key)
                if cb is None:
                    continue
            grid_layout.addWidget(cb, visible_idx // 4, visible_idx % 4)
            visible_idx += 1

        root_layout.addLayout(grid_layout)

        self.update_all_toggle_text()
        return panel

    def toggle_section_visibility(self, key, is_visible):
        widget = self.section_widgets.get(key)
        if widget is not None:
            widget.setVisible(is_visible)
        cb = self.section_checkboxes.get(key)
        if cb is not None:
            cb.setChecked(is_visible)
        self._update_washer_control_checkbox()
        self.update_all_toggle_text()

    def toggle_all_sections(self):
        should_show = not all(cb.isChecked() for cb in self.section_checkboxes.values())
        for key, cb in self.section_checkboxes.items():
            cb.blockSignals(True)
            cb.setChecked(should_show)
            cb.blockSignals(False)
            if key in self.section_widgets:
                self.section_widgets[key].setVisible(should_show)
        self._update_washer_control_checkbox()
        self.update_all_toggle_text()

    def toggle_washer_control_sections(self, checked):
        for key in self.washer_control_section_keys:
            cb = self.section_checkboxes.get(key)
            widget = self.section_widgets.get(key)
            if cb is not None:
                cb.blockSignals(True)
                cb.setChecked(checked)
                cb.blockSignals(False)
                if widget is not None:
                    widget.setVisible(checked)
        self.update_all_toggle_text()

    def _update_washer_control_checkbox(self):
        if not hasattr(self, 'washer_control_cb'):
            return
        child_checked = []
        for key in self.washer_control_section_keys:
            cb = self.section_checkboxes.get(key)
            if cb is not None:
                child_checked.append(cb.isChecked())
        if not child_checked:
            self.washer_control_cb.setChecked(False)
            return
        self.washer_control_cb.blockSignals(True)
        self.washer_control_cb.setChecked(all(child_checked))
        self.washer_control_cb.blockSignals(False)

    def update_all_toggle_text(self):
        if not self.section_checkboxes:
            self.all_toggle_btn.setText('All select')
            return
        self._update_washer_control_checkbox()
        all_checked = all(cb.isChecked() for cb in self.section_checkboxes.values())
        self.all_toggle_btn.setText('All deselect' if all_checked else 'All select')


    def initUI(self):
        self.main_layout = QVBoxLayout()
        self.main_layout.setAlignment(Qt.AlignTop)
        self.main_layout.setContentsMargins(8, 8, 8, 8)
        self.main_layout.setSpacing(0)

        # --- Port 설정 ---
        self.port_layout = QHBoxLayout()
        self.port_lb = self.make_label('Shell Port')
        self.port_layout.addWidget(self.port_lb)

        self.port_cb = QComboBox()
        self.port_cb.setStyleSheet('color: black; background: white')
        self.port_cb.setFixedWidth(DEFAULT_BUTTON_WIDTH)
        self.refresh_ports()
        self.port_cb.currentTextChanged.connect(self.Port_onChanged)
        self.port_layout.addWidget(self.port_cb)

        self.refresh_bt = self.make_button('&Refresh', bgcolor="#e0e0e0", handler=self.refresh_callback)
        self.port_layout.addWidget(self.refresh_bt)

        self.conn_bt = self.make_button('&Connect', bgcolor="#e0e0e0", handler=self.connect_callback)
        self.port_layout.addWidget(self.conn_bt)

        self.disconnect_bt = self.make_button('&Disconnect', bgcolor="#e0e0e0", handler=self.port_disconnect_callback)
        self.port_layout.addWidget(self.disconnect_bt)
        self.port_section = self.make_section_widget(self.port_layout)

        # --- Relay 설정 ---
        self.relay_layout = QHBoxLayout()
        self.relay_lb = self.make_label('Relay Port')
        self.relay_layout.addWidget(self.relay_lb)

        self.lcp_port_cb = QComboBox()
        self.lcp_port_cb.setStyleSheet('color: black; background: white')
        self.lcp_port_cb.currentTextChanged.connect(self.LCP_Port_onChanged)
        self.relay_layout.addWidget(self.lcp_port_cb)

        self.osp_port_cb = QComboBox()
        self.osp_port_cb.setStyleSheet('color: black; background: white')
        self.osp_port_cb.currentTextChanged.connect(self.OSP_Port_onChanged)
        self.relay_layout.addWidget(self.osp_port_cb)

        self.refresh_relay_ports()

        self.relay_bt = self.make_button('&Connect', handler=self.relay_callback)
        self.relay_layout.addWidget(self.relay_bt)

        self.relay_disconnect_bt = self.make_button('&Disconnect', handler=self.relay_disconnect_callback)
        self.relay_disconnect_bt.setEnabled(False)
        self.relay_disconnect_bt.setStyleSheet('background:#b0b0b0;')
        self.relay_layout.addWidget(self.relay_disconnect_bt)
        self.relay_section = self.make_section_widget(self.relay_layout)

        # --- 세탁기 제어 ---
        self.washer_cont_layout = QHBoxLayout()
        self.washer_cont_lb = self.make_label('Washer Control')
        self.washer_cont_layout.addWidget(self.washer_cont_lb)

        self.washer_power_bt = self.make_button('&On / Off', handler=self.washer_power_process)
        self.washer_cont_layout.addWidget(self.washer_power_bt)

        self.washer_start_bt = self.make_button('&Start / Stop', handler=self.washer_start_process)
        self.washer_cont_layout.addWidget(self.washer_start_bt)
        self.washer_cont_layout.addStretch()
        self.washer_cont_section = self.make_section_widget(self.washer_cont_layout)

        # --- Display Control ---
        self.disp_cont_layout = QHBoxLayout()
        self.disp_cont_lb = self.make_label('Display Control')
        self.disp_cont_layout.addWidget(self.disp_cont_lb)

        self.disp_on_bt = self.make_button('&Display On', handler=self.disp_on_process)
        self.disp_cont_layout.addWidget(self.disp_on_bt)

        self.disp_off_bt = self.make_button('&Display Off', handler=self.disp_off_process)
        self.disp_cont_layout.addWidget(self.disp_off_bt)

        self.disp_bl_bt = self.make_button('&Backlight On', handler=self.disp_bl_process)
        self.disp_cont_layout.addWidget(self.disp_bl_bt)
        self.disp_cont_layout.addStretch()
        self.disp_cont_section = self.make_section_widget(self.disp_cont_layout)

        # --- touch ---
        self.touch_layout = QHBoxLayout()
        self.touch_layout.setContentsMargins(0, 0, 0, 0)
        self.touch_lb = self.make_label('Touch')
        self.touch_layout.addWidget(self.touch_lb)

        self.left_touch_lb = self.make_label('L(x,y)', width=38)
        self.touch_layout.addWidget(self.left_touch_lb)
        coord_validator = QIntValidator(0, 10000, self)
        self.left_touch_x_le = QLineEdit('77')
        self.left_touch_x_le.setFixedWidth(30)
        self.left_touch_x_le.setValidator(coord_validator)
        self.left_touch_x_le.setStyleSheet('color: black;')
        self.left_touch_x_le.setPlaceholderText('x')
        self.left_touch_y_le = QLineEdit('318')
        self.left_touch_y_le.setFixedWidth(30)
        self.left_touch_y_le.setValidator(coord_validator)
        self.left_touch_y_le.setStyleSheet('color: black;')
        self.left_touch_y_le.setPlaceholderText('y')
        self.touch_layout.addWidget(self.left_touch_x_le)
        self.touch_layout.addWidget(self.left_touch_y_le)

        self.right_touch_lb = self.make_label('R(x,y)', width=38)
        self.touch_layout.addWidget(self.right_touch_lb)
        self.right_touch_x_le = QLineEdit('923')
        self.right_touch_x_le.setFixedWidth(30)
        self.right_touch_x_le.setValidator(coord_validator)
        self.right_touch_x_le.setStyleSheet('color: black;')
        self.right_touch_x_le.setPlaceholderText('x')
        self.right_touch_y_le = QLineEdit('318')
        self.right_touch_y_le.setFixedWidth(30)
        self.right_touch_y_le.setValidator(coord_validator)
        self.right_touch_y_le.setStyleSheet('color: black;')
        self.right_touch_y_le.setPlaceholderText('y')
        self.touch_layout.addWidget(self.right_touch_x_le)
        self.touch_layout.addWidget(self.right_touch_y_le)
        self.touch_layout.addStretch()

        self.l_touch_bt = self.make_button('&Left', width=71, handler=self.left_touch)
        self.touch_layout.addWidget(self.l_touch_bt)

        self.r_touch_bt = self.make_button('&Right', width=71, handler=self.right_touch)
        self.touch_layout.addWidget(self.r_touch_bt)

        self.auto_touch_bt = self.make_button('&Auto touch', width=72, handler=self.toggle_auto_touch_process)
        self.touch_layout.addWidget(self.auto_touch_bt)
        self.touch_section = self.make_section_widget(self.touch_layout)

        # --- App Start ---
        self.app_start_layout = QHBoxLayout()
        self.app_start_layout.setContentsMargins(0, 0, 0, 0)
        self.app_start_layout.setSpacing(self.touch_layout.spacing() if hasattr(self, 'touch_layout') else 6)
        self.app_start_lb = self.make_label('App Start')
        self.app_start_layout.addWidget(self.app_start_lb)

        self.app_add_bt = self.make_button('&Add (max:4)', handler=self.add_app)
        self.app_start_layout.addWidget(self.app_add_bt)
        self.app_delete_bt = self.make_button('&Delete', handler=self.delete_app)
        self.app_start_layout.addWidget(self.app_delete_bt)
        self.app_all_start_bt = self.make_button('&All Start', handler=self.app_all_start_process)
        self.app_start_layout.addWidget(self.app_all_start_bt)
        self.app_all_stop_bt = self.make_button('&All Stop', handler=lambda: self.stop_repeat_mode())
        self.app_start_layout.addWidget(self.app_all_stop_bt)

        self.app_rows_layout = QVBoxLayout()
        self.app_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.app_rows_layout.setSpacing(4)
        self.app_row_widgets = [None] * self.max_apps
        self.app_inputs = [None] * self.max_apps

        for idx in range(self.max_apps):
            self._create_app_row(idx)

        self.app_active_count = 1
        for idx in range(self.max_apps):
            visible = idx < self.app_active_count
            self.app_row_widgets[idx].setVisible(visible)

        self.app_start_body_layout = QVBoxLayout()
        self.app_start_body_layout.setContentsMargins(0, 0, 0, 0)
        self.app_start_body_layout.setSpacing(4)
        self.app_start_body_layout.addLayout(self.app_start_layout)
        self.app_start_body_layout.addLayout(self.app_rows_layout)
        self.app_start_section = self.make_section_widget(self.app_start_body_layout)

        # --- Watchdog Box ---
        self.watchdog_layout = QHBoxLayout()
        self.watchdog_layout.setContentsMargins(0, 0, 0, 0)
        self.watchdog_lb = self.make_label('Watchdog')
        self.watchdog_lb.setFixedHeight(50)
        self.watchdog_layout.addWidget(self.watchdog_lb)

        self.watchdog_tb = QTextBrowser()
        self.watchdog_tb.setStyleSheet("background-color: #222; color: #ff0;")
        self.watchdog_tb.setFixedHeight(50)
        self.watchdog_tb.setMaximumWidth(500)
        self.watchdog_tb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.watchdog_layout.addWidget(self.watchdog_tb)
        self.watchdog_section = self.make_section_widget(self.watchdog_layout)
        self.watchdog_section.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # --- Memory Tracking Box ---
        self.meminfo_layout = QHBoxLayout()
        self.meminfo_layout.setContentsMargins(0, 0, 0, 0)
        self.meminfo_bt = self.make_button('&Memory Check', width=DEFAULT_LABEL_WIDTH, handler=self.memory_check)
        self.meminfo_bt.setFixedHeight(50)
        self.meminfo_layout.addWidget(self.meminfo_bt)

        self.meminfo_tb = QTextBrowser()
        self.meminfo_tb.setStyleSheet("background-color: #222; color: #ff0;")
        self.meminfo_tb.setFixedHeight(50)
        self.meminfo_tb.setMaximumWidth(500)
        self.meminfo_tb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.meminfo_layout.addWidget(self.meminfo_tb)
        self.meminfo_section = self.make_section_widget(self.meminfo_layout)
        self.meminfo_section.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # --- TFLite Parser ---
        self.tflite_layout = QHBoxLayout()
        self.tflite_layout.setContentsMargins(0, 0, 0, 0)
        self.tflite_lb = self.make_label('TFLite Parser')
        self.tflite_layout.addWidget(self.tflite_lb)

        self.tflite_parse_bt = self.make_button('Parse .tflite', width=100, handler=self.parse_tflite_model_dialog)
        self.tflite_layout.addWidget(self.tflite_parse_bt)
        self.tflite_batch_bt = self.make_button('Batch Folder', width=100, handler=self.batch_parse_tflite_folder_dialog)
        self.tflite_layout.addWidget(self.tflite_batch_bt)
        self.tflite_fit_bt = self.make_button('Fit Timing', width=100, handler=self.fit_timing_from_csv_dialog)
        self.tflite_layout.addWidget(self.tflite_fit_bt)
        self.tflite_layout.addStretch()
        self.tflite_section = self.make_section_widget(self.tflite_layout)

        # --- Shell ---
        self.shell_input_layout = QHBoxLayout()
        self.shell_lb = self.make_label('Shell')
        self.shell_input_layout.addWidget(self.shell_lb)

        self.shell_le = HistoryLineEdit()
        self.shell_le.setStyleSheet('color: black;')
        self.shell_le.setMinimumWidth(100)
        self.shell_le.textChanged[str].connect(self.Shell_onChanged)
        self.shell_le.returnPressed.connect(self.shell_le_entered)
        self.shell_input_layout.addWidget(self.shell_le)

        self.shell_clear_bt = self.make_button('Clear', handler=self.shell_clear, width=50)
        self.shell_input_layout.addWidget(self.shell_clear_bt)
        # shell response 영역의 shell_section 하단에 로그 표시 창을 추가함.
        self.shell_resp_layout = QHBoxLayout()

        self.shell_tb = QTextBrowser()
        self.shell_tb.setStyleSheet("background-color: black; color: white;")
        self.shell_tb.setAcceptRichText(True)
        self.shell_tb.setOpenExternalLinks(True)
        self.shell_tb.setMinimumHeight(600)
        self.shell_tb.setMinimumWidth(600)
        self.shell_section_layout = QVBoxLayout()
        self.shell_section_layout.addLayout(self.shell_input_layout)
        self.shell_section_layout.addWidget(self.shell_tb)
        self.shell_section_layout.setContentsMargins(0, 0, 0, 0)
        self.shell_section = self.make_section_widget(self.shell_section_layout)

        # --- CMD ---
        self.cmd_layout = QHBoxLayout()
        self.cmd_lb = self.make_label('Windows CMD')
        self.cmd_layout.addWidget(self.cmd_lb)

        self.cmd_le = HistoryLineEdit()
        self.cmd_le.returnPressed.connect(self.send_cmd_input)
        self.cmd_le.setMinimumWidth(400)
        self.cmd_layout.addWidget(self.cmd_le)

        self.cmd_clear_bt = self.make_button('Clear', handler=self.cmd_clear, width=50)
        self.cmd_layout.addWidget(self.cmd_clear_bt)

        self.cmd_tb = QTextBrowser()
        self.cmd_tb.setStyleSheet("background-color: #222; color: #eee;")
        self.cmd_tb.setMinimumHeight(300)
        self.cmd_section_layout = QVBoxLayout()
        self.cmd_section_layout.addLayout(self.cmd_layout)
        self.cmd_section_layout.addWidget(self.cmd_tb)
        self.cmd_section_layout.setContentsMargins(0, 0, 0, 0)
        self.cmd_section = self.make_section_widget(self.cmd_section_layout)

        # --- cmd.exe start ---
        self.cmd_process = QProcess(self)
        self.cmd_process.setProcessChannelMode(QProcess.MergedChannels)
        self.cmd_process.readyReadStandardOutput.connect(self.read_cmd_output)
        self.cmd_process.start("cmd.exe")

        self.visibility_widget = self.create_visibility_control_widget([
            ('relay', 'Relay Port', self.relay_section),
            ('washer', 'Washer Control', self.washer_cont_section),
            ('display', 'Display Control', self.disp_cont_section),
            ('touch', 'Touch', self.touch_section),
            ('app_start', 'App Start', self.app_start_section),
            ('watchdog', 'Watchdog', self.watchdog_section),
            ('memory', 'Memory Check', self.meminfo_section),
            ('tflite', 'TFLite Parser', self.tflite_section),
            ('cmd', 'Windows CMD', self.cmd_section),
        ])

        self.left_layout = QVBoxLayout()
        self.left_layout.setAlignment(Qt.AlignTop)
        self.left_layout.setContentsMargins(0, 0, 0, 0)
        self.left_layout.setSpacing(4)
        self.left_layout.addWidget(self.visibility_widget)
        self.left_layout.addWidget(self.port_section)
        self.left_layout.addWidget(self.relay_section)
        self.left_layout.addWidget(self.washer_cont_section)
        self.left_layout.addWidget(self.disp_cont_section)
        self.left_layout.addWidget(self.touch_section)
        self.left_layout.addWidget(self.app_start_section)
        self.left_layout.addWidget(self.watchdog_section)
        self.left_layout.addWidget(self.meminfo_section)
        self.left_layout.addWidget(self.tflite_section)
        self.left_layout.addWidget(self.cmd_section)

        self.left_panel = QWidget()
        self.left_panel.setLayout(self.left_layout)
        self.left_panel.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum)
        self.left_panel.setMinimumWidth(0)

        self.right_layout = QVBoxLayout()
        self.right_layout.setAlignment(Qt.AlignTop)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.setSpacing(0)
        self.right_layout.addWidget(self.shell_section)

        self.top_layout = QHBoxLayout()
        self.top_layout.setAlignment(Qt.AlignTop)
        self.top_layout.setContentsMargins(0, 0, 0, 0)
        self.top_layout.setSpacing(8)
        self.top_layout.addWidget(self.left_panel, 0, Qt.AlignTop)
        self.top_layout.addLayout(self.right_layout, stretch=1)
        self.main_layout.addLayout(self.top_layout)

        self.setLayout(self.main_layout)
        self.setWindowTitle('Welcome to LUPA AI')
        self.setGeometry(300, 200, 1000, 700)
        self.setStyleSheet("background-color: #f5f5f5;")
        self.set_width_combobox(DEFAULT_BUTTON_WIDTH)
        self.set_buttons(False)

        self.show()

    # --- app start function ---
    def _create_app_row(self, idx):
        row_widget = QWidget()
        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(self.app_start_layout.spacing() if hasattr(self, 'app_start_layout') else 4)

        row_lb = self.make_label(f'App #{idx + 1}')
        row_layout.addWidget(row_lb)

        app_input = QLineEdit()
        app_input.setPlaceholderText('multi_ai_demo1')
        # Match the width occupied by Add + Delete (+ one layout gap) so Start aligns with All Start.
        app_input.setFixedWidth((DEFAULT_BUTTON_WIDTH * 2) + row_layout.spacing())
        row_layout.addWidget(app_input)
        self.app_inputs[idx] = app_input

        start_bt = self.make_button('Start', handler=lambda _, app_idx=idx: self.toggle_app_running(app_idx))
        repeat_bt = self.make_button('Repeat', handler=lambda _, app_idx=idx: self.toggle_repeat_mode(app_idx))

        self.app_start_bts[idx] = start_bt
        self.start_repeat_bts[idx] = repeat_bt
        self._update_repeat_button(idx)

        row_layout.addWidget(start_bt)
        row_layout.addWidget(repeat_bt)
        row_widget.setLayout(row_layout)
        self.app_rows_layout.addWidget(row_widget)
        self.app_row_widgets[idx] = row_widget

    def add_app(self):
        if self.app_active_count >= self.max_apps:
            self.shell_tb.append("앱은 최대 4개까지만 지원됩니다.")
            return
        self.app_row_widgets[self.app_active_count].setVisible(True)
        self.app_active_count += 1
        self._update_all_app_controls_state()

    def delete_app(self):
        if self.app_active_count <= 1:
            return
        idx = self.app_active_count - 1
        if self.repeat_flags[idx]:
            self.stop_repeat_mode(idx=idx)
        if self.is_app_running[idx]:
            self.app_quit_process(idx, True)
        self._clear_app_slot(idx)
        self.app_row_widgets[idx].setVisible(False)
        self.app_active_count -= 1
        self._update_all_app_controls_state()

    def _clear_app_slot(self, idx):
        if idx < 0 or idx >= self.max_apps:
            return
        self.app_names[idx] = None
        self.is_app_running[idx] = False
        self.repeat_flags[idx] = False
        self.repeat_counts[idx] = 0
        self.repeat_timers[idx].stop()
        if self.app_inputs[idx] is not None:
            self.app_inputs[idx].clear()
        if self.app_start_bts[idx] is not None:
            self.app_start_bts[idx].setText('S&tart')
            self.app_start_bts[idx].setStyleSheet('background-color: #e0e0e0; color: black;')
            self.app_start_bts[idx].setEnabled(self._buttons_enabled and not self.repeat_flags[idx])
        self._update_repeat_button(idx)

    def _active_app_indices(self):
        return list(range(self.app_active_count))

    def _repeat_button_text(self, idx):
        if self.repeat_flags[idx]:
            return f'Stop ({self.repeat_counts[idx]})'
        return 'Repeat'

    def _repeat_button_style(self, idx):
        if self.repeat_flags[idx]:
            return 'background:#ffd966; color:black;'
        if self._buttons_enabled:
            return 'background-color: #e0e0e0; color: black;'
        return 'background-color: #b0b0b0;'

    def _app_input_to_fullname(self, idx, silent=False):
        if idx < 0 or idx >= self.max_apps or self.app_inputs[idx] is None:
            return ''
        raw_name = self.app_inputs[idx].text().strip()
        full_name = self.get_full_app_name(raw_name)
        if not full_name and not silent:
            self.shell_tb.append(f"App #{idx + 1}의 packagename을 입력하세요.")
        return full_name

    def _update_repeat_button(self, idx):
        bt = self.start_repeat_bts[idx]
        if bt is None:
            return
        visible = idx < self.app_active_count
        bt.setText(self._repeat_button_text(idx))
        bt.setStyleSheet(self._repeat_button_style(idx))
        bt.setEnabled(self._buttons_enabled and visible)

    def _update_all_app_controls_state(self):
        self.app_add_bt.setEnabled(self._buttons_enabled and self.app_active_count < self.max_apps)
        self.app_delete_bt.setEnabled(self._buttons_enabled and self.app_active_count > 1)
        self.app_all_start_bt.setEnabled(self._buttons_enabled)
        self.app_all_stop_bt.setEnabled(self._buttons_enabled)
        for idx in range(self.max_apps):
            active_row = idx < self.app_active_count
            for btn in (self.app_start_bts[idx], self.start_repeat_bts[idx]):
                if btn is None:
                    continue
                btn.setVisible(active_row)
                if btn is self.app_start_bts[idx]:
                    btn.setText('St&op' if self.is_app_running[idx] else 'S&tart')
                    btn.setEnabled(self._buttons_enabled and active_row and not self.repeat_flags[idx])
                    if not self._buttons_enabled or self.repeat_flags[idx]:
                        btn.setStyleSheet('background-color: #b0b0b0;')
                    elif self.is_app_running[idx]:
                        btn.setStyleSheet('background:#feffcd; color:black;')
                    else:
                        btn.setStyleSheet('background-color: #e0e0e0; color: black;')
                else:
                    self._update_repeat_button(idx)
            if self.app_inputs[idx] is not None:
                self.app_inputs[idx].setEnabled(self._buttons_enabled)
                self.app_inputs[idx].setVisible(active_row)

        for idx in range(self.max_apps):
            self.app_row_widgets[idx].setVisible(idx < self.app_active_count)

    def update_all_app_buttons_enabled(self):
        self._update_all_app_controls_state()

    def app_start_process(self, idx):
        if idx >= self.max_apps or idx >= self.app_active_count:
            return
        full_name = self._app_input_to_fullname(idx)
        if not full_name:
            return
        self.app_names[idx] = full_name
        self.shell_cmd(f"app start {full_name}")
        self.is_app_running[idx] = True
        self._update_all_app_controls_state()
        self.app_inputs[idx].setToolTip(full_name)

    def toggle_app_running(self, idx):
        if idx >= self.max_apps or idx >= self.app_active_count:
            return
        if self.is_app_running[idx]:
            self.app_quit_process(idx, True)
        else:
            self.app_start_process(idx)

    def app_all_start_process(self):
        for delay, idx in enumerate(self._active_app_indices()):
            if not self.is_app_running[idx]:
                QTimer.singleShot(delay * 500, lambda idx=idx: self.app_start_process(idx))

    # --- app quit function ---
    def app_quit_process(self, idx, mode):
        if idx >= self.max_apps or idx >= self.app_active_count:
            return
        full_name = self._app_input_to_fullname(idx, silent=True)
        if not full_name:
            full_name = self.app_names[idx]
        if not full_name:
            return
        self.shell_cmd(f"app quit {full_name}")
        if mode and self.app_start_bts[idx] is not None:
            self.app_start_bts[idx].setText('S&tart')
            self.app_start_bts[idx].setEnabled(self._buttons_enabled and not self.repeat_flags[idx] and idx < self.app_active_count)
            if self._buttons_enabled and not self.repeat_flags[idx]:
                self.app_start_bts[idx].setStyleSheet('background-color: #e0e0e0; color: black;')
            else:
                self.app_start_bts[idx].setStyleSheet('background-color: #b0b0b0;')
        self.is_app_running[idx] = False
        self._update_all_app_controls_state()

    def app_all_quit_process(self, mode):
        for delay, idx in enumerate(self._active_app_indices()):
            if self.is_app_running[idx]:
                QTimer.singleShot(delay * 500, lambda idx=idx: self.app_quit_process(idx, mode))

    # --- repeat start function ---
    def toggle_repeat_mode(self, idx):
        if idx >= self.max_apps or idx >= self.app_active_count:
            return
        if self.repeat_flags[idx]:
            self.stop_repeat_mode(idx=idx)
        else:
            self.start_repeat_mode(idx)

    def start_repeat_mode(self, idx):
        if idx >= self.max_apps or idx >= self.app_active_count:
            return
        if not self._app_input_to_fullname(idx):
            return
        self.repeat_flags[idx] = True
        self.repeat_counts[idx] = 1
        self.app_start_process(idx)
        self._update_repeat_button(idx)
        self.meminfo_tb.clear()
        self.watchdog_tb.clear()
        self.watchdog_lb.setStyleSheet('color:black; background:#e0e0e0;')
        self.repeat_timers[idx].start(DEFAULT_REPEAT_INTERVAL, self)

    def stop_repeat_mode(self, idx=None):
        targets = self._active_app_indices() if idx is None else [idx]
        for target_idx in targets:
            if target_idx >= self.max_apps or not self.app_names[target_idx] or not self.repeat_flags[target_idx]:
                continue
            self.repeat_flags[target_idx] = False
            self.repeat_timers[target_idx].stop()
            self.repeat_counts[target_idx] = 0
            self._update_repeat_button(target_idx)
            self.app_quit_process(target_idx, True)
        if idx is None:
            self.app_all_quit_process(True)
            self.last_meminfo_value = 0

    # --- CMD function ---
    def send_cmd_input(self):
        cmd = self.cmd_le.text().strip() + '\n'
        self.cmd_process.write(cmd.encode('utf-8'))
        self.cmd_le.clear()

    def read_cmd_output(self):
        output = self.cmd_process.readAllStandardOutput().data().decode('cp949', errors='replace')
        self.cmd_tb.append(output)

    def cmd_clear(self):
        self.cmd_tb.clear()

    # --- TFLite parser ---
    def parse_tflite_model_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            'Select TFLite model',
            '',
            'TensorFlow Lite Model (*.tflite);;All Files (*)'
        )
        if not file_path:
            return

        try:
            analysis = self.analyze_tflite_model(file_path)
            report = self.format_tflite_analysis_report(analysis)
            summary_csv_path, layers_csv_path = self.write_model_analysis_csvs(analysis)
        except Exception as e:
            self.shell_tb.append(f"[TFLite Parser] {e}")
            return

        self.shell_tb.append(f"[TFLite Parser] Parsed: {file_path}")
        self.shell_tb.append(f"[TFLite Parser] Summary CSV: {summary_csv_path}")
        self.shell_tb.append(f"[TFLite Parser] Layer CSV: {layers_csv_path}")
        self.shell_tb.append("[TFLite Parser] Fill measured_time_ms in the summary CSV to use Fit Timing.")
        self.shell_tb.append(f"<pre style='margin:0'>{html.escape(report)}</pre>")

    def build_tflite_report(self, file_path):
        analysis = self.analyze_tflite_model(file_path)
        return self.format_tflite_analysis_report(analysis)

    def batch_parse_tflite_folder_dialog(self):
        folder_path = QFileDialog.getExistingDirectory(self, 'Select folder with .tflite models', '')
        if not folder_path:
            return

        model_paths = self._collect_tflite_files(folder_path)
        if not model_paths:
            self.shell_tb.append(f"[TFLite Parser] No .tflite files found in: {folder_path}")
            return

        analyses = []
        failures = []
        for model_path in model_paths:
            try:
                analyses.append(self.analyze_tflite_model(model_path))
            except Exception as e:
                failures.append((model_path, str(e)))

        if not analyses:
            self.shell_tb.append(f"[TFLite Parser] Failed to analyze all models in: {folder_path}")
            for model_path, error_text in failures:
                self.shell_tb.append(f"[TFLite Parser] {model_path} -> {error_text}")
            return

        summary_csv_path, layers_csv_path = self.write_batch_analysis_csvs(analyses, folder_path)
        self.shell_tb.append(f"[TFLite Parser] Batch analyzed: {len(analyses)} model(s)")
        self.shell_tb.append(f"[TFLite Parser] Batch summary CSV: {summary_csv_path}")
        self.shell_tb.append(f"[TFLite Parser] Batch layer CSV: {layers_csv_path}")
        self.shell_tb.append("[TFLite Parser] Fill measured_time_ms in the batch summary CSV to use Fit Timing.")
        if failures:
            self.shell_tb.append(f"[TFLite Parser] Failed models: {len(failures)}")
            for model_path, error_text in failures[:10]:
                self.shell_tb.append(f"[TFLite Parser] {model_path} -> {error_text}")

    def fit_timing_from_csv_dialog(self):
        csv_path, _ = QFileDialog.getOpenFileName(
            self,
            'Select timing CSV',
            '',
            'CSV Files (*.csv);;All Files (*)'
        )
        if not csv_path:
            return

        try:
            fit_result = self.fit_timing_model_from_csv(csv_path)
            coeff_csv_path, pred_csv_path = self.write_timing_fit_csvs(fit_result, csv_path)
        except Exception as e:
            self.shell_tb.append(f"[TFLite Parser] {e}")
            return

        self.shell_tb.append(f"[TFLite Parser] Timing fit source: {csv_path}")
        self.shell_tb.append(f"[TFLite Parser] Coefficients CSV: {coeff_csv_path}")
        self.shell_tb.append(f"[TFLite Parser] Predictions CSV: {pred_csv_path}")
        self.shell_tb.append(f"<pre style='margin:0'>{html.escape(self.format_timing_fit_report(fit_result))}</pre>")

    def analyze_tflite_model(self, file_path):
        with open(file_path, 'rb') as model_file:
            data = model_file.read()

        if len(data) < 8:
            raise ValueError('Invalid TFLite file: file is too small.')
        if data[4:8] != TFLITE_FILE_IDENTIFIER:
            raise ValueError('Invalid TFLite file: missing TFL3 identifier.')

        root_offset = struct.unpack_from('<I', data, 0)[0]
        model = FlatBufferTable(data, root_offset)
        operator_codes = [self._parse_tflite_operator_code(item) for item in model.get_table_vector(1)]
        subgraphs = model.get_table_vector(2)
        if not subgraphs:
            raise ValueError('This TFLite model does not contain any subgraph.')

        subgraph = subgraphs[0]
        tensors = [self._parse_tflite_tensor(item) for item in subgraph.get_table_vector(0)]
        operators = subgraph.get_table_vector(3)
        graph_inputs = [idx for idx in subgraph.get_int_vector(1) if idx >= 0]
        graph_outputs = [idx for idx in subgraph.get_int_vector(2) if idx >= 0]

        analysis = {
            'model_name': os.path.basename(file_path),
            'model_path': file_path,
            'schema_version': model.get_uint32(0, 0),
            'subgraph_name': subgraph.get_string(4, 'main'),
            'operators_total': len(operators),
            'graph_inputs_text': self._format_tensor_list(graph_inputs, tensors),
            'graph_outputs_text': self._format_tensor_list(graph_outputs, tensors),
            'layers': [],
            'totals': {
                'supported_layers': 0,
                'params': 0,
                'mul': 0,
                'add': 0,
                'total': 0,
                'mac': 0,
                'output_elements': 0,
            },
            'per_op': {},
        }

        for layer_idx, operator in enumerate(operators):
            opcode_index = operator.get_uint32(0, 0)
            op_info = operator_codes[opcode_index] if opcode_index < len(operator_codes) else {
                'name': f'OPCODE_{opcode_index}',
                'version': 1,
            }
            input_indices = [idx for idx in operator.get_int_vector(1) if idx >= 0]
            output_indices = [idx for idx in operator.get_int_vector(2) if idx >= 0]
            layer_cost = self._estimate_layer_ops(op_info['name'], input_indices, output_indices, tensors)

            output_dtype = '-'
            if output_indices and 0 <= output_indices[0] < len(tensors):
                output_dtype = tensors[output_indices[0]].get('type_name', '-')

            layer_entry = {
                'layer_index': layer_idx,
                'op_name': op_info['name'],
                'version': op_info['version'],
                'inputs_text': self._format_tensor_list(input_indices, tensors),
                'outputs_text': self._format_tensor_list(output_indices, tensors),
                'output_dtype': output_dtype,
                'supported': layer_cost is not None,
                'params': layer_cost['params'] if layer_cost is not None else 0,
                'mul': layer_cost['mul'] if layer_cost is not None else 0,
                'add': layer_cost['add'] if layer_cost is not None else 0,
                'total': layer_cost['total'] if layer_cost is not None else 0,
                'mac': layer_cost['mac'] if layer_cost is not None else 0,
                'output_elements': layer_cost['output_elements'] if layer_cost is not None else 0,
                'detail': (
                    layer_cost['detail']
                    if layer_cost is not None
                    else 'unsupported or insufficient shape information'
                ),
            }
            analysis['layers'].append(layer_entry)

            if layer_cost is None:
                continue

            totals = analysis['totals']
            totals['supported_layers'] += 1
            totals['params'] += layer_cost['params']
            totals['mul'] += layer_cost['mul']
            totals['add'] += layer_cost['add']
            totals['total'] += layer_cost['total']
            totals['mac'] += layer_cost['mac']
            totals['output_elements'] += layer_cost['output_elements']

            op_totals = analysis['per_op'].setdefault(op_info['name'], {
                'layers': 0,
                'params': 0,
                'mul': 0,
                'add': 0,
                'total': 0,
                'mac': 0,
                'output_elements': 0,
            })
            op_totals['layers'] += 1
            op_totals['params'] += layer_cost['params']
            op_totals['mul'] += layer_cost['mul']
            op_totals['add'] += layer_cost['add']
            op_totals['total'] += layer_cost['total']
            op_totals['mac'] += layer_cost['mac']
            op_totals['output_elements'] += layer_cost['output_elements']

        return analysis

    def format_tflite_analysis_report(self, analysis):
        lines = [
            f"Model: {analysis['model_name']}",
            f"Schema version: {analysis['schema_version']}",
            f"Subgraph: {analysis['subgraph_name']}",
            f"Operators: {analysis['operators_total']}",
            f"Graph inputs: {analysis['graph_inputs_text']}",
            f"Graph outputs: {analysis['graph_outputs_text']}",
            "",
        ]

        for layer in analysis['layers']:
            lines.append(f"[Layer {layer['layer_index']:03d}] {layer['op_name']} (v{layer['version']})")
            lines.append(f"  inputs : {layer['inputs_text']}")
            lines.append(f"  outputs: {layer['outputs_text']}")
            if layer['supported']:
                lines.append(
                    "  stats  : "
                    f"params={layer['params']}, "
                    f"mul={layer['mul']}, add={layer['add']}, "
                    f"total={layer['total']}, mac={layer['mac']}, "
                    f"output_elements={layer['output_elements']}"
                )
            else:
                lines.append("  stats  : unavailable (unsupported op or incomplete shape information)")
            lines.append(f"  detail : {layer['detail']}")
            lines.append("")

        totals = analysis['totals']
        lines.append(
            "Supported summary: "
            f"layers={totals['supported_layers']}/{analysis['operators_total']}, "
            f"params={totals['params']}, mul={totals['mul']}, add={totals['add']}, "
            f"total={totals['total']}, mac={totals['mac']}, "
            f"output_elements={totals['output_elements']}"
        )
        for op_name in ('FULLY_CONNECTED', 'CONV_2D', 'DEPTHWISE_CONV_2D'):
            op_totals = analysis['per_op'].get(op_name)
            if not op_totals:
                continue
            lines.append(
                f"{op_name} summary: "
                f"layers={op_totals['layers']}, params={op_totals['params']}, "
                f"mul={op_totals['mul']}, add={op_totals['add']}, "
                f"total={op_totals['total']}, mac={op_totals['mac']}, "
                f"output_elements={op_totals['output_elements']}"
            )
        lines.append("Supported ops: FULLY_CONNECTED, CONV_2D, DEPTHWISE_CONV_2D")
        lines.append("Note: counts are estimated from tensor shapes only. For FLOAT tensors this can be read as float ops; for quantized tensors it is arithmetic op count. Activation, requantization, memory traffic, and framework overhead are not included.")
        return '\n'.join(lines)

    def write_model_analysis_csvs(self, analysis):
        base_path = os.path.splitext(analysis['model_path'])[0]
        summary_csv_path = base_path + '_summary.csv'
        layers_csv_path = base_path + '_layers.csv'
        self._write_summary_csv(summary_csv_path, [analysis])
        self._write_layers_csv(layers_csv_path, [analysis])
        return summary_csv_path, layers_csv_path

    def write_batch_analysis_csvs(self, analyses, folder_path):
        summary_csv_path = os.path.join(folder_path, 'tflite_batch_summary.csv')
        layers_csv_path = os.path.join(folder_path, 'tflite_batch_layers.csv')
        self._write_summary_csv(summary_csv_path, analyses)
        self._write_layers_csv(layers_csv_path, analyses)
        return summary_csv_path, layers_csv_path

    def _write_summary_csv(self, csv_path, analyses):
        headers = [
            'model_name',
            'model_path',
            'schema_version',
            'subgraph_name',
            'operators_total',
            'supported_layers',
            'total_params',
            'total_mul',
            'total_add',
            'total_ops',
            'total_mac',
            'total_output_elements',
            'fc_layers',
            'fc_mac',
            'conv_layers',
            'conv_mac',
            'depthwise_layers',
            'depthwise_mac',
            'measured_time_ms',
        ]
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=headers)
            writer.writeheader()
            for analysis in analyses:
                writer.writerow(self._analysis_summary_row(analysis))

    def _write_layers_csv(self, csv_path, analyses):
        headers = [
            'model_name',
            'model_path',
            'layer_index',
            'op_name',
            'version',
            'output_dtype',
            'supported',
            'params',
            'mul',
            'add',
            'total_ops',
            'mac',
            'output_elements',
            'inputs',
            'outputs',
            'detail',
        ]
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=headers)
            writer.writeheader()
            for analysis in analyses:
                for layer in analysis['layers']:
                    writer.writerow({
                        'model_name': analysis['model_name'],
                        'model_path': analysis['model_path'],
                        'layer_index': layer['layer_index'],
                        'op_name': layer['op_name'],
                        'version': layer['version'],
                        'output_dtype': layer['output_dtype'],
                        'supported': int(layer['supported']),
                        'params': layer['params'],
                        'mul': layer['mul'],
                        'add': layer['add'],
                        'total_ops': layer['total'],
                        'mac': layer['mac'],
                        'output_elements': layer['output_elements'],
                        'inputs': layer['inputs_text'],
                        'outputs': layer['outputs_text'],
                        'detail': layer['detail'],
                    })

    def _analysis_summary_row(self, analysis):
        totals = analysis['totals']
        fc_totals = analysis['per_op'].get('FULLY_CONNECTED', {})
        conv_totals = analysis['per_op'].get('CONV_2D', {})
        depthwise_totals = analysis['per_op'].get('DEPTHWISE_CONV_2D', {})
        return {
            'model_name': analysis['model_name'],
            'model_path': analysis['model_path'],
            'schema_version': analysis['schema_version'],
            'subgraph_name': analysis['subgraph_name'],
            'operators_total': analysis['operators_total'],
            'supported_layers': totals['supported_layers'],
            'total_params': totals['params'],
            'total_mul': totals['mul'],
            'total_add': totals['add'],
            'total_ops': totals['total'],
            'total_mac': totals['mac'],
            'total_output_elements': totals['output_elements'],
            'fc_layers': fc_totals.get('layers', 0),
            'fc_mac': fc_totals.get('mac', 0),
            'conv_layers': conv_totals.get('layers', 0),
            'conv_mac': conv_totals.get('mac', 0),
            'depthwise_layers': depthwise_totals.get('layers', 0),
            'depthwise_mac': depthwise_totals.get('mac', 0),
            'measured_time_ms': '',
        }

    def _collect_tflite_files(self, folder_path):
        model_paths = []
        for root_dir, _, filenames in os.walk(folder_path):
            for filename in filenames:
                if filename.lower().endswith('.tflite'):
                    model_paths.append(os.path.join(root_dir, filename))
        model_paths.sort()
        return model_paths

    def fit_timing_model_from_csv(self, csv_path):
        with open(csv_path, 'r', newline='', encoding='utf-8-sig') as csv_file:
            reader = csv.DictReader(csv_file)
            rows = list(reader)

        if not rows:
            raise ValueError('Timing CSV is empty.')

        samples = []
        skipped_rows = []
        for row_index, row in enumerate(rows, start=2):
            measured_time = self._parse_optional_float(row.get('measured_time_ms'))
            if measured_time is None:
                skipped_rows.append((row_index, 'missing measured_time_ms'))
                continue

            feature_row = self._extract_timing_feature_row(row)
            if feature_row is None:
                skipped_rows.append((row_index, 'missing model_path or feature columns'))
                continue

            sample = {
                'row_index': row_index,
                'model_name': row.get('model_name') or os.path.basename(row.get('model_path', '')),
                'model_path': row.get('model_path', ''),
                'measured_time_ms': measured_time,
            }
            sample.update(feature_row)
            samples.append(sample)

        if len(samples) < 2:
            raise ValueError('Need at least 2 timing samples with measured_time_ms to fit coefficients.')

        feature_defs = [
            ('fc_mac_m', 'fc_mac / 1e6'),
            ('conv_mac_m', 'conv_mac / 1e6'),
            ('depthwise_mac_m', 'depthwise_mac / 1e6'),
            ('total_add_m', 'total_add / 1e6'),
            ('total_output_elements_m', 'total_output_elements / 1e6'),
        ]
        active_features = [
            (key, label) for key, label in feature_defs
            if any(abs(sample[key]) > 0 for sample in samples)
        ]
        if not active_features:
            raise ValueError('No non-zero timing features found in CSV.')

        x_rows = []
        y_values = []
        for sample in samples:
            x_rows.append([1.0] + [sample[key] for key, _ in active_features])
            y_values.append(sample['measured_time_ms'])

        coefficients = self._solve_ridge_regression(x_rows, y_values, ridge_lambda=1e-9)
        predictions = []
        for sample, x_row in zip(samples, x_rows):
            predicted = sum(coef * value for coef, value in zip(coefficients, x_row))
            residual = sample['measured_time_ms'] - predicted
            prediction_row = dict(sample)
            prediction_row['predicted_time_ms'] = predicted
            prediction_row['residual_ms'] = residual
            predictions.append(prediction_row)

        mae = sum(abs(row['residual_ms']) for row in predictions) / len(predictions)
        rmse = (
            sum((row['residual_ms'] ** 2) for row in predictions) / len(predictions)
        ) ** 0.5
        y_mean = sum(y_values) / len(y_values)
        ss_tot = sum((value - y_mean) ** 2 for value in y_values)
        ss_res = sum((row['residual_ms'] ** 2) for row in predictions)
        r2 = None if ss_tot == 0 else 1.0 - (ss_res / ss_tot)

        return {
            'source_csv': csv_path,
            'samples': predictions,
            'active_features': active_features,
            'coefficients': coefficients,
            'metrics': {
                'sample_count': len(predictions),
                'skipped_count': len(skipped_rows),
                'mae_ms': mae,
                'rmse_ms': rmse,
                'r2': r2,
            },
            'skipped_rows': skipped_rows,
        }

    def format_timing_fit_report(self, fit_result):
        coeffs = fit_result['coefficients']
        active_features = fit_result['active_features']
        metrics = fit_result['metrics']
        terms = [f"{coeffs[0]:.6f}"]
        for coef, (_, label) in zip(coeffs[1:], active_features):
            sign = '+' if coef >= 0 else '-'
            terms.append(f"{sign} {abs(coef):.6f} * ({label})")

        lines = [
            f"Timing fit source: {fit_result['source_csv']}",
            f"Samples used: {metrics['sample_count']}",
            f"Rows skipped: {metrics['skipped_count']}",
            f"Formula (ms): time ~= {' '.join(terms)}",
            f"MAE: {metrics['mae_ms']:.6f} ms",
            f"RMSE: {metrics['rmse_ms']:.6f} ms",
        ]
        if metrics['r2'] is not None:
            lines.append(f"R^2: {metrics['r2']:.6f}")
        else:
            lines.append("R^2: unavailable (all measured times are identical)")
        if fit_result['skipped_rows']:
            preview = ', '.join(f"row {row_idx}" for row_idx, _ in fit_result['skipped_rows'][:10])
            lines.append(f"Skipped rows: {preview}")
        return '\n'.join(lines)

    def write_timing_fit_csvs(self, fit_result, source_csv_path):
        base_path = os.path.splitext(source_csv_path)[0]
        coeff_csv_path = base_path + '_timing_fit_coefficients.csv'
        pred_csv_path = base_path + '_timing_fit_predictions.csv'

        with open(coeff_csv_path, 'w', newline='', encoding='utf-8-sig') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['term', 'coefficient_ms'])
            writer.writerow(['intercept', fit_result['coefficients'][0]])
            for coef, (feature_key, label) in zip(fit_result['coefficients'][1:], fit_result['active_features']):
                writer.writerow([feature_key, coef])
            writer.writerow([])
            writer.writerow(['metric', 'value'])
            for metric_name, metric_value in fit_result['metrics'].items():
                writer.writerow([metric_name, metric_value])

        pred_headers = [
            'row_index',
            'model_name',
            'model_path',
            'measured_time_ms',
            'predicted_time_ms',
            'residual_ms',
            'fc_mac',
            'conv_mac',
            'depthwise_mac',
            'total_add',
            'total_output_elements',
        ]
        with open(pred_csv_path, 'w', newline='', encoding='utf-8-sig') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=pred_headers)
            writer.writeheader()
            for sample in fit_result['samples']:
                writer.writerow({
                    'row_index': sample['row_index'],
                    'model_name': sample['model_name'],
                    'model_path': sample['model_path'],
                    'measured_time_ms': sample['measured_time_ms'],
                    'predicted_time_ms': sample['predicted_time_ms'],
                    'residual_ms': sample['residual_ms'],
                    'fc_mac': int(round(sample['fc_mac_m'] * 1000000.0)),
                    'conv_mac': int(round(sample['conv_mac_m'] * 1000000.0)),
                    'depthwise_mac': int(round(sample['depthwise_mac_m'] * 1000000.0)),
                    'total_add': int(round(sample['total_add_m'] * 1000000.0)),
                    'total_output_elements': int(round(sample['total_output_elements_m'] * 1000000.0)),
                })

        return coeff_csv_path, pred_csv_path

    def _extract_timing_feature_row(self, row):
        feature_names = ['fc_mac', 'conv_mac', 'depthwise_mac', 'total_add', 'total_output_elements']
        if all((str(row.get(name, '')).strip() != '') for name in feature_names):
            return {
                'fc_mac_m': (self._parse_optional_float(row.get('fc_mac')) or 0.0) / 1000000.0,
                'conv_mac_m': (self._parse_optional_float(row.get('conv_mac')) or 0.0) / 1000000.0,
                'depthwise_mac_m': (self._parse_optional_float(row.get('depthwise_mac')) or 0.0) / 1000000.0,
                'total_add_m': (self._parse_optional_float(row.get('total_add')) or 0.0) / 1000000.0,
                'total_output_elements_m': (self._parse_optional_float(row.get('total_output_elements')) or 0.0) / 1000000.0,
            }

        model_path = row.get('model_path', '').strip()
        if not model_path or not os.path.exists(model_path):
            return None
        analysis = self.analyze_tflite_model(model_path)
        summary_row = self._analysis_summary_row(analysis)
        return {
            'fc_mac_m': float(summary_row['fc_mac']) / 1000000.0,
            'conv_mac_m': float(summary_row['conv_mac']) / 1000000.0,
            'depthwise_mac_m': float(summary_row['depthwise_mac']) / 1000000.0,
            'total_add_m': float(summary_row['total_add']) / 1000000.0,
            'total_output_elements_m': float(summary_row['total_output_elements']) / 1000000.0,
        }

    def _parse_optional_float(self, value):
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _solve_ridge_regression(self, x_rows, y_values, ridge_lambda=1e-9):
        size = len(x_rows[0])
        normal_matrix = [[0.0 for _ in range(size)] for _ in range(size)]
        rhs = [0.0 for _ in range(size)]

        for x_row, y_value in zip(x_rows, y_values):
            for i in range(size):
                rhs[i] += x_row[i] * y_value
                for j in range(size):
                    normal_matrix[i][j] += x_row[i] * x_row[j]

        for i in range(1, size):
            normal_matrix[i][i] += ridge_lambda

        return self._solve_linear_system(normal_matrix, rhs)

    def _solve_linear_system(self, matrix, rhs):
        size = len(rhs)
        aug = [row[:] + [rhs_value] for row, rhs_value in zip(matrix, rhs)]

        for col in range(size):
            pivot_row = max(range(col, size), key=lambda row_idx: abs(aug[row_idx][col]))
            if abs(aug[pivot_row][col]) < 1e-18:
                raise ValueError('Failed to fit timing coefficients: singular matrix.')
            if pivot_row != col:
                aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

            pivot = aug[col][col]
            for j in range(col, size + 1):
                aug[col][j] /= pivot

            for row_idx in range(size):
                if row_idx == col:
                    continue
                factor = aug[row_idx][col]
                if factor == 0:
                    continue
                for j in range(col, size + 1):
                    aug[row_idx][j] -= factor * aug[col][j]

        return [aug[row_idx][size] for row_idx in range(size)]

    def _parse_tflite_operator_code(self, operator_code_table):
        if operator_code_table.has_field(3):
            builtin_code = operator_code_table.get_int32(3, 0)
        else:
            builtin_code = operator_code_table.get_uint8(0, 0)
        custom_code = operator_code_table.get_string(1, '')
        version = operator_code_table.get_int32(2, 1)
        if custom_code:
            name = f"CUSTOM({custom_code})"
        else:
            name = TFLITE_BUILTIN_OP_NAMES.get(builtin_code, f"BUILTIN_{builtin_code}")
        return {
            'builtin_code': builtin_code,
            'custom_code': custom_code,
            'version': version,
            'name': name,
        }

    def _parse_tflite_tensor(self, tensor_table):
        shape = tensor_table.get_int_vector(0)
        shape_signature = tensor_table.get_int_vector(7)
        tensor_type = tensor_table.get_uint8(1, 0)
        return {
            'name': tensor_table.get_string(3, '(unnamed)'),
            'shape': shape if shape else shape_signature,
            'buffer': tensor_table.get_uint32(2, 0),
            'type': tensor_type,
            'type_name': TFLITE_TENSOR_TYPE_NAMES.get(tensor_type, f"TYPE_{tensor_type}"),
        }

    def _format_tensor_list(self, tensor_indices, tensors):
        if not tensor_indices:
            return '-'
        return ' | '.join(self._format_tensor_entry(idx, tensors) for idx in tensor_indices)

    def _format_tensor_entry(self, tensor_index, tensors):
        if tensor_index < 0 or tensor_index >= len(tensors):
            return f"#{tensor_index} <invalid>"
        tensor_info = tensors[tensor_index]
        return (
            f"#{tensor_index} {tensor_info['name']} "
            f"{self._format_shape(tensor_info['shape'])} {tensor_info['type_name']}"
        )

    def _format_shape(self, shape):
        if shape is None:
            return '?'
        if len(shape) == 0:
            return '[]'
        return '[' + ', '.join(str(dim) for dim in shape) + ']'

    def _shape_numel(self, shape):
        if shape is None:
            return None
        total = 1
        for dim in shape:
            if dim is None or dim < 0:
                return None
            total *= dim
        return total

    def _estimate_layer_ops(self, op_name, input_indices, output_indices, tensors):
        estimators = {
            'FULLY_CONNECTED': self._estimate_fc_ops,
            'CONV_2D': self._estimate_conv2d_ops,
            'DEPTHWISE_CONV_2D': self._estimate_depthwise_conv2d_ops,
        }
        estimator = estimators.get(op_name)
        if estimator is None:
            return None
        return estimator(input_indices, output_indices, tensors)

    def _estimate_fc_ops(self, input_indices, output_indices, tensors):
        if len(input_indices) < 2 or not output_indices:
            return None

        input_tensor = tensors[input_indices[0]] if input_indices[0] < len(tensors) else None
        weight_tensor = tensors[input_indices[1]] if input_indices[1] < len(tensors) else None
        output_tensor = tensors[output_indices[0]] if output_indices[0] < len(tensors) else None
        bias_tensor = None
        if len(input_indices) > 2 and input_indices[2] < len(tensors):
            bias_tensor = tensors[input_indices[2]]

        if input_tensor is None or weight_tensor is None or output_tensor is None:
            return None

        weight_shape = weight_tensor['shape']
        output_shape = output_tensor['shape']
        input_shape = input_tensor['shape']

        accumulation_depth = None
        if weight_shape and len(weight_shape) >= 2 and weight_shape[-1] > 0:
            accumulation_depth = weight_shape[-1]

        output_elements = self._shape_numel(output_shape)
        if output_elements is None and weight_shape and len(weight_shape) >= 2:
            units = weight_shape[0] if weight_shape[0] > 0 else None
            input_elements = self._shape_numel(input_shape)
            if units and accumulation_depth and input_elements and input_elements % accumulation_depth == 0:
                output_elements = (input_elements // accumulation_depth) * units

        if accumulation_depth is None or output_elements is None:
            return None

        units = None
        if weight_shape and len(weight_shape) >= 1 and weight_shape[0] > 0:
            units = weight_shape[0]

        weight_params = self._shape_numel(weight_shape)
        bias_params = self._shape_numel(bias_tensor['shape']) if bias_tensor is not None else 0
        mul_count = output_elements * accumulation_depth
        add_count = output_elements * max(accumulation_depth - 1, 0)
        if bias_tensor is not None:
            add_count += output_elements

        return {
            'params': (weight_params or 0) + (bias_params or 0),
            'units': units if units is not None else 0,
            'input_depth': accumulation_depth,
            'has_bias': bias_tensor is not None,
            'mul': mul_count,
            'add': add_count,
            'mac': mul_count,
            'total': mul_count + add_count,
            'output_elements': output_elements,
            'detail': (
                f"FC units={units if units is not None else '?'}, "
                f"input_depth={accumulation_depth}, "
                f"output_elements={output_elements}, "
                f"bias={'yes' if bias_tensor is not None else 'no'}"
            ),
        }

    def _estimate_conv2d_ops(self, input_indices, output_indices, tensors):
        if len(input_indices) < 2 or not output_indices:
            return None

        input_tensor = tensors[input_indices[0]] if input_indices[0] < len(tensors) else None
        weight_tensor = tensors[input_indices[1]] if input_indices[1] < len(tensors) else None
        output_tensor = tensors[output_indices[0]] if output_indices[0] < len(tensors) else None
        bias_tensor = None
        if len(input_indices) > 2 and input_indices[2] < len(tensors):
            bias_tensor = tensors[input_indices[2]]

        if input_tensor is None or weight_tensor is None or output_tensor is None:
            return None

        input_shape = input_tensor['shape']
        weight_shape = weight_tensor['shape']
        output_shape = output_tensor['shape']
        if not weight_shape or len(weight_shape) < 4:
            return None

        output_elements = self._shape_numel(output_shape)
        if output_elements is None:
            return None

        out_channels = weight_shape[0]
        kernel_h = weight_shape[1]
        kernel_w = weight_shape[2]
        in_channels = weight_shape[3]
        if min(out_channels, kernel_h, kernel_w, in_channels) < 0:
            return None

        accumulation_depth = kernel_h * kernel_w * in_channels
        weight_params = self._shape_numel(weight_shape)
        bias_params = self._shape_numel(bias_tensor['shape']) if bias_tensor is not None else 0
        mul_count = output_elements * accumulation_depth
        add_count = output_elements * max(accumulation_depth - 1, 0)
        if bias_tensor is not None:
            add_count += output_elements

        input_channels = input_shape[-1] if input_shape and len(input_shape) >= 1 else in_channels
        return {
            'params': (weight_params or 0) + (bias_params or 0),
            'mul': mul_count,
            'add': add_count,
            'mac': mul_count,
            'total': mul_count + add_count,
            'output_elements': output_elements,
            'detail': (
                f"Conv2D kernel={kernel_h}x{kernel_w}, "
                f"in_ch={input_channels}, out_ch={out_channels}, "
                f"output_elements={output_elements}, "
                f"bias={'yes' if bias_tensor is not None else 'no'}"
            ),
        }

    def _estimate_depthwise_conv2d_ops(self, input_indices, output_indices, tensors):
        if len(input_indices) < 2 or not output_indices:
            return None

        input_tensor = tensors[input_indices[0]] if input_indices[0] < len(tensors) else None
        weight_tensor = tensors[input_indices[1]] if input_indices[1] < len(tensors) else None
        output_tensor = tensors[output_indices[0]] if output_indices[0] < len(tensors) else None
        bias_tensor = None
        if len(input_indices) > 2 and input_indices[2] < len(tensors):
            bias_tensor = tensors[input_indices[2]]

        if input_tensor is None or weight_tensor is None or output_tensor is None:
            return None

        input_shape = input_tensor['shape']
        weight_shape = weight_tensor['shape']
        output_shape = output_tensor['shape']
        if not weight_shape or len(weight_shape) < 4:
            return None

        output_elements = self._shape_numel(output_shape)
        if output_elements is None:
            return None

        kernel_h = weight_shape[1]
        kernel_w = weight_shape[2]
        out_channels = weight_shape[3]
        input_channels = input_shape[-1] if input_shape and len(input_shape) >= 1 else None
        if input_channels is None or input_channels <= 0 or min(kernel_h, kernel_w, out_channels) < 0:
            return None

        depth_multiplier = out_channels // input_channels if input_channels > 0 else 0
        accumulation_depth = kernel_h * kernel_w
        weight_params = self._shape_numel(weight_shape)
        bias_params = self._shape_numel(bias_tensor['shape']) if bias_tensor is not None else 0
        mul_count = output_elements * accumulation_depth
        add_count = output_elements * max(accumulation_depth - 1, 0)
        if bias_tensor is not None:
            add_count += output_elements

        return {
            'params': (weight_params or 0) + (bias_params or 0),
            'mul': mul_count,
            'add': add_count,
            'mac': mul_count,
            'total': mul_count + add_count,
            'output_elements': output_elements,
            'detail': (
                f"DepthwiseConv2D kernel={kernel_h}x{kernel_w}, "
                f"in_ch={input_channels}, out_ch={out_channels}, "
                f"depth_multiplier={depth_multiplier}, "
                f"output_elements={output_elements}, "
                f"bias={'yes' if bias_tensor is not None else 'no'}"
            ),
        }

    # --- Layer options ---
    def set_width_combobox(self, size):
        combos = [self.port_cb, self.lcp_port_cb, self.osp_port_cb]
        for combo in combos:
            combo.setMaximumWidth(size)

    def set_buttons(self, set_enabled=True):
        self._buttons_enabled = set_enabled
        buttons = [
            self.disconnect_bt,
            self.washer_power_bt,
            self.washer_start_bt,
            self.disp_on_bt,
            self.disp_off_bt,
            self.disp_bl_bt,
            self.app_all_start_bt,
            self.app_all_stop_bt,
            self.r_touch_bt,
            self.l_touch_bt,
            self.auto_touch_bt,
            self.meminfo_bt,
            self.app_add_bt,
            self.app_delete_bt
        ]
        for button in buttons:
            button.setEnabled(set_enabled)
            if set_enabled:
                button.setStyleSheet('background-color: #e0e0e0; color: black;')
            else:
                button.setStyleSheet('background-color: #b0b0b0;')

        for btn in self.app_start_bts:
            if btn is not None:
                btn.setEnabled(set_enabled)
                btn.setStyleSheet('background-color: #e0e0e0; color: black;' if set_enabled else 'background-color: #b0b0b0;')
        for idx in range(self.max_apps):
            self._update_repeat_button(idx)
        self._update_all_app_controls_state()

    def refresh_callback(self):
        self.refresh_ports()
        self.refresh_relay_ports()

    def shell_clear(self):
        self.shell_tb.clear()

    def shell_le_entered(self):
        cmd = self.shell_le.text().strip()
        self.shell_cmd(cmd)
        self.shell_le.clear()

    # --- Shell functions ---
    def shell_cmd(self, cmd):
        self.last_shell_cmd = cmd
        if self.log_reader and self.log_reader.serial_conn and self.log_reader.serial_conn.is_open:
            try:
                self.log_reader.serial_conn.write((cmd + '\n').encode('utf-8'))
            except Exception as e:
                self.shell_tb.append(f"{e}")
        else:
            self.shell_tb.append("Port is not connected.")

    def get_port_list(self):
        return [port.device for port in serial.tools.list_ports.comports()]

    def refresh_ports(self):
        self.port_cb.clear()
        port_list = self.get_port_list()
        self.port_cb.addItems(port_list)
        self.Port_Text = port_list[0] if port_list else None

    # --- Relay ports ---
    def refresh_relay_ports(self):
        port_list = self.get_port_list()
        self.lcp_port_cb.clear()
        self.osp_port_cb.clear()
        self.lcp_port_cb.addItems(port_list)
        self.osp_port_cb.addItems(port_list)
        if port_list:
            self.lcp_port = port_list[0]
            self.osp_port = port_list[0]
        else:
            self.lcp_port = None
            self.osp_port = None

    def Port_onChanged(self, text):
        self.Port_Text = text

    def LCP_Port_onChanged(self, text):
        self.lcp_port = text

    def OSP_Port_onChanged(self, text):
        self.osp_port = text

    def relay_callback(self):
        if self.relay_worker:
            self.relay_worker.stop()
            self.relay_worker = None
        try:
            self.relay_worker = RelayWorker(self.lcp_port, self.osp_port)
            self.relay_worker.error_occurred.connect(self.handle_relay_serial_error)
            self.relay_worker.start()
            self.relay_bt.setText('Connecting ...')
            self.relay_bt.setStyleSheet('background:#afcfee; color:black;')
            self.shell_tb.append(f"Trying to connect Relay Port... {self.lcp_port} <-> {self.osp_port}")
            self.relay_bt.setEnabled(False)
            self.relay_connecting = True
            self.relay_error = False
            QTimer.singleShot(1000, self.check_relay_success)
            lcp_idx = self.port_cb.findText(self.lcp_port)
            osp_idx = self.port_cb.findText(self.osp_port)
            for idx in sorted([lcp_idx, osp_idx], reverse=True):
                if idx != -1:
                    self.port_cb.removeItem(idx)
        except Exception as e:
            self.handle_relay_serial_error(str(e))

    def check_relay_success(self):
        if getattr(self, 'relay_connecting', False) and not getattr(self, 'relay_error', False) and self.relay_worker is not None and self.relay_worker.isRunning():
            self.shell_tb.append("Relay Port connected successfully!")
            self.relay_bt.setStyleSheet('background:#feffcd; color:black;')
            self.relay_bt.setText('&Connected')
            self.relay_connecting = False
            self.relay_disconnect_bt.setEnabled(True)
            self.relay_disconnect_bt.setStyleSheet('background-color: #e0e0e0; color: black;')

    def handle_relay_serial_error(self, msg):
        self.relay_bt.setStyleSheet('background:red; color:white;')
        self.relay_bt.setText('&Failed to Connect')
        self.shell_tb.append(f"{msg}")
        self.relay_connecting = False
        self.relay_error = True
        if self.relay_worker:
            self.relay_worker.stop()
            self.relay_worker = None
        QTimer.singleShot(3000, self.reset_relay_connect_button)

    def reset_relay_connect_button(self):
        self.relay_bt.setText('&Connect')
        self.relay_bt.setStyleSheet('background-color: #e0e0e0; color: black;')
        self.relay_bt.setEnabled(True)

    def relay_disconnect_callback(self):
        if self.relay_worker:
            self.relay_worker.stop()
            self.relay_worker = None
            self.relay_bt.setStyleSheet('background-color: #e0e0e0; color: black;')
            self.relay_bt.setText('&Connect')
            self.relay_bt.setEnabled(True)
            self.relay_disconnect_bt.setEnabled(False)
            self.relay_disconnect_bt.setStyleSheet('background:#b0b0b0;')
            self.shell_tb.append("Relay port disconnected!")
            self.refresh_callback()
        else:
            self.shell_tb.append("Relay port already disconnected")

    def Shell_onChanged(self, text):
        self.Shell_CMD = text

    # --- Shell port ---
    def connect_callback(self):
        if self.log_reader:
            self.log_reader.stop()
            self.log_reader = None
        try:
            self.log_reader = SerialLogReader(self.Port_Text)
            self.log_reader.log_received.connect(self.handle_shell_log)
            self.log_reader.error_occurred.connect(self.handle_serial_error)
            self.log_reader.start()
            self.conn_bt.setText('Connecting ...')
            self.conn_bt.setStyleSheet('background:#afcfee; color:black;')
            self.conn_bt.setEnabled(False)
            self.conn_connecting = True
            self.conn_error = False
            self.shell_tb.append(f"Connecting ... {self.Port_Text}")
            idx = self.port_cb.findText(self.Port_Text)
            if idx != -1:
                self.lcp_port_cb.removeItem(idx)
                self.osp_port_cb.removeItem(idx)
            QTimer.singleShot(1000, self.check_connect_success)
        except Exception as e:
            self.conn_bt.setStyleSheet('background:red; color:white;')
            self.shell_tb.append(f"[Connect Error] {e}")
            self.handle_serial_error(str(e))

    def check_connect_success(self):
        if getattr(self, 'conn_connecting', False) and not getattr(self, 'conn_error', False) and self.log_reader is not None and self.log_reader.isRunning():
            self.shell_tb.append("Port connected successfully!")
            self.conn_bt.setStyleSheet('background:#feffcd; color:black;')
            self.conn_bt.setText('Connected')
            self.conn_connecting = False
            self.set_buttons(True)

    def handle_serial_error(self, msg):
        self.conn_bt.setStyleSheet('background:red; color:white;')
        self.set_buttons(False)
        self.conn_bt.setText('Failed to Connect')
        if self.relay_worker:
            self.relay_worker.stop()
            self.relay_worker = None
        QTimer.singleShot(3000, self.reset_connect_button)
        self.port_disconnect_callback()

    def reset_connect_button(self):
        self.conn_bt.setText('Connect')
        self.conn_bt.setStyleSheet('background-color: #e0e0e0; color: black;')
        self.conn_bt.setEnabled(True)

    def port_disconnect_callback(self):
        self.stop_repeat_mode()
        self.app_all_quit_process(False)
        self.refresh_callback()
        self.conn_bt.setStyleSheet('background:#e0e0e0; color:black;')
        self.conn_bt.setEnabled(True)
        self.conn_bt.setText('Connect')
        self.shell_tb.append("Shell port disconnected!")
        if self.log_reader:
            QTimer.singleShot(50, self.log_reader.stop)
            self.log_reader = None
            self.set_buttons(False)

    # --- Shell log handling ---
    def handle_shell_log(self, text):
        if self.shell_tb.document().blockCount() >= DEFAULT_MAX_LOG_LINES:
            try:
                time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"log_{time}.txt"
                with open(filename, "a", encoding="utf-8") as f:
                    f.write(self.shell_tb.toPlainText())
                    f.write("\n---[AUTO CLEARED]---\n")
            except Exception as e:
                print(f"Log save failed: {e}")
            self.shell_tb.clear()

        self.shell_tb.append(text)

        if "wdg_svc_monitor_thread: 3" in text:
            self.watchdog_lb.setStyleSheet('color:white; background:red;')
            lines = text.splitlines()
            for idx, line in enumerate(lines):
                if "wdg_svc_monitor_thread: 3" in line:
                    start = max(0, idx - 3)
                    end = min(len(lines), idx + 4)
                    context = lines[start:end]
                    msg = "<br>".join(context)
                    self.watchdog_tb.append(f"<span style='color:#f44;background:#222'><b>[Watchdog Detected]</b><br>{msg}</span>")

        if self.mem_check:
            import re
            match = re.search(r"(\d+)\s*/\s*16777216", text)
            if match:
                new_value = int(match.group(1))
                if self.last_meminfo_value != new_value:
                    diff = new_value - self.last_meminfo_value
                    if self.last_meminfo_value == 0:
                        msg = f"<b>MMB Usage</b> : {new_value}"
                    else:
                        msg = f"<b>MMB Usage</b> : {self.last_meminfo_value} → {new_value} ({'+' if diff > 0 else ''}{diff})"
                    self.meminfo_tb.append(msg)
                self.last_meminfo_value = new_value

        if "Permission denied, please try again." in text:
            self.resend_cmd = self.last_shell_cmd
            self.shell_cmd("@lupa123")
            QTimer.singleShot(500, self.resend_last_cmd)

    def resend_last_cmd(self):
        if hasattr(self, 'resend_cmd') and self.resend_cmd:
            self.shell_cmd(self.resend_cmd)
            self.resend_cmd = None

    def memory_info(self):
        self.shell_tb.clear()
        self.shell_cmd("meminfo")

    def memory_check(self):
        if self.mem_check:
            self.mem_check = False
            self.meminfo_bt.setStyleSheet('background:#e0e0e0; color: black;')
        else:
            self.mem_check = True
            self.meminfo_bt.setStyleSheet('background:#feffcd; color:black;')

    def app_all_delete_process(self):
        for idx in self._active_app_indices():
            app_name = self._app_input_to_fullname(idx, silent=True)
            if app_name:
                self.shell_cmd(f"pkg uninstall {app_name}")

    def _parse_touch_int(self, raw_text, default):
        try:
            return int(raw_text.strip())
        except (TypeError, ValueError):
            return default

    def _touch_point(self, side):
        defaults = {'left': (77, 318), 'right': (923, 318)}
        default_x, default_y = defaults.get(side, (0, 0))
        if side == 'left':
            x = self._parse_touch_int(self.left_touch_x_le.text(), default_x)
            y = self._parse_touch_int(self.left_touch_y_le.text(), default_y)
        else:
            x = self._parse_touch_int(self.right_touch_x_le.text(), default_x)
            y = self._parse_touch_int(self.right_touch_y_le.text(), default_y)
        return x, y

    def left_touch(self):
        x, y = self._touch_point('left')
        self.shell_cmd(f"input tab {x} {y}")

    def right_touch(self):
        x, y = self._touch_point('right')
        self.shell_cmd(f"input tab {x} {y}")

    def toggle_auto_touch_process(self):
        if self.auto_touch_running:
            self.auto_touch_stop_process()
        else:
            self.auto_touch_process()

    def auto_touch_process(self):
        self.shell_tb.append("Auto touch start.")
        self.auto_touch_running = True
        self.auto_touch_bt.setText('Auto &Stop')
        self.auto_touch_bt.setStyleSheet('background:#ffd966; color:black;')
        self.touch_timer.start(DEFAULT_TOUCH_INTERVAL, self)

    def auto_touch_stop_process(self):
        self.shell_tb.append("Auto touch stopped.")
        self.auto_touch_running = False
        self.auto_touch_bt.setText('&Auto')
        self.auto_touch_bt.setStyleSheet('background:#e0e0e0; color: black;')
        self.touch_timer.stop()

    def touch_timer_callback(self):
        x, y = self._touch_point('right')
        self.shell_cmd(f"input tab {x} {y}")

    def timerEvent(self, event):
        for idx, timer in enumerate(self.repeat_timers):
            if event.timerId() == timer.timerId():
                if self.is_app_running[idx]:
                    self.app_quit_process(idx, True)
                    if self.mem_check and all(not running for running in self.is_app_running):
                        QTimer.singleShot(500, self.memory_info)
                else:
                    self.repeat_counts[idx] += 1
                    self.app_start_process(idx)
                    self._update_repeat_button(idx)
                return
        if event.timerId() == self.touch_timer.timerId():
            self.touch_timer_callback()

    def washer_power_process(self):
        self.shell_cmd("settings set volatile.sh.knob \"key p\"")

    def washer_start_process(self):
        self.shell_cmd("settings set volatile.sh.knob \"key s\"")

    def disp_on_process(self):
        self.shell_cmd("settings set volatile.ev.power.osp_state 2")

    def disp_off_process(self):
        self.shell_cmd("settings set volatile.ev.power.osp_state 0")

    def disp_bl_process(self):
        self.shell_cmd("bl set 255")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    my_app = MyApp()
    sys.exit(app.exec_())
