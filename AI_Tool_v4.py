import sys
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
DEMO_APPS = [
    "lupa.usr.multi_ai_demo1",  # App #1
    "lupa.usr.multi_ai_demo2",  # App #2
    "lupa.usr.multi_ai_demo3",  # App #3
    "lupa.usr.multi_ai_demo4"   # App #4
    ]

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

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.app_start_bts = []
        self.app_quit_bts = []
        self.start_repeat_bts = []
        self.is_app_running = [False] * 4
        self.repeat_flags = [False] * 4
        self.repeat_timers = [QBasicTimer() for _ in range(4)]
        self.repeat_counts = [0] * 4
        self.last_meminfo_value = 0
        self.touch_timer = QBasicTimer()
        self.log_reader = None
        self.relay_worker = None
        self.mem_check = False
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

    def make_dummy_button(self, width=DEFAULT_BUTTON_WIDTH):
        btn = QPushButton('')
        btn.setFixedWidth(width)
        btn.setStyleSheet('background-color:transparent')
        return btn

    def initUI(self):
        self.main_layout = QVBoxLayout()

        # --- Port 설정 ---
        self.port_layout = QHBoxLayout()
        self.port_lb = self.make_label('Shell Port')
        self.port_layout.addWidget(self.port_lb)

        self.port_cb = QComboBox()
        self.port_cb.setStyleSheet('color: black; background: white')
        self.refresh_ports()
        self.port_cb.currentTextChanged.connect(self.Port_onChanged)
        self.port_layout.addWidget(self.port_cb)

        self.refresh_bt = self.make_button('&Refresh', bgcolor="#e0e0e0", handler=self.refresh_callback)
        self.port_layout.addWidget(self.refresh_bt)

        self.conn_bt = self.make_button('&Connect', bgcolor="#e0e0e0", handler=self.connect_callback)
        self.port_layout.addWidget(self.conn_bt)

        self.disconnect_bt = self.make_button('&Disconnect', bgcolor="#e0e0e0", handler=self.port_disconnect_callback)
        self.port_layout.addWidget(self.disconnect_bt)

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

        # --- 세탁기 Control ---
        self.washer_cont_layout = QHBoxLayout()
        self.washer_cont_lb = self.make_label('세탁기 행정')
        self.washer_cont_layout.addWidget(self.washer_cont_lb)

        self.washer_power_bt = self.make_button('&On / Off', handler=self.washer_power_process)
        self.washer_cont_layout.addWidget(self.washer_power_bt)

        self.washer_start_bt = self.make_button('&Start / Stop', handler=self.washer_start_process)
        self.washer_cont_layout.addWidget(self.washer_start_bt)

        self.washer_cont_layout.addWidget(self.make_dummy_button())
        self.washer_cont_layout.addWidget(self.make_dummy_button())

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

        self.disp_cont_layout.addWidget(self.make_dummy_button())
        
        # --- touch ---
        self.touch_layout = QHBoxLayout()
        self.touch_lb = self.make_label('Touch')
        self.touch_layout.addWidget(self.touch_lb)

        self.l_touch_bt = self.make_button('&Left', handler=self.left_touch)
        self.touch_layout.addWidget(self.l_touch_bt)

        self.r_touch_bt = self.make_button('&Right', handler=self.right_touch)
        self.touch_layout.addWidget(self.r_touch_bt)

        self.auto_touch_bt = self.make_button('&Auto', handler=self.auto_touch_process)
        self.touch_layout.addWidget(self.auto_touch_bt)

        self.auto_touch_stop_bt = self.make_button('&Auto Stop', handler=self.auto_touch_stop_process)
        self.touch_layout.addWidget(self.auto_touch_stop_bt)

        # --- app start ---
        self.app_start_layout = QHBoxLayout()
        self.app_start_lb = self.make_label('App Start')
        self.app_start_layout.addWidget(self.app_start_lb)
        for i in range(4):
            bt = self.make_button(f'&App #{i+1}', width=81, handler=lambda _, idx=i: self.app_start_process(idx))
            self.app_start_bts.append(bt)
            self.app_start_layout.addWidget(bt)
        self.app_all_start_bt = self.make_button('&All Start', handler=self.app_all_start_process)
        self.app_start_layout.addWidget(self.app_all_start_bt)

        # --- app quit ---
        self.app_quit_layout = QHBoxLayout()
        self.app_quit_lb = self.make_label('App Quit')
        self.app_quit_layout.addWidget(self.app_quit_lb)
        for i in range(4):
            bt = self.make_button(f'&App #{i+1}', width=81, handler=lambda _, idx=i: self.app_quit_process(idx, True))
            self.app_quit_bts.append(bt)
            self.app_quit_layout.addWidget(bt)
        self.app_all_quit_bt = self.make_button('&All Quit', handler=lambda: self.app_all_quit_process(True))
        self.app_quit_layout.addWidget(self.app_all_quit_bt)

        # --- test layer ---
        self.test_layer = QHBoxLayout()
        self.test_lb = self.make_label('Repeat Test')
        self.test_layer.addWidget(self.test_lb)
        for i in range(4):
            bt = self.make_button(f'&App #{i+1}', width=81, handler=lambda _, idx=i: self.start_repeat_mode(idx))
            self.start_repeat_bts.append(bt)
            self.test_layer.addWidget(bt)
        self.stop_repeat_bt = self.make_button('&Stop Repeat', handler=lambda: self.stop_repeat_mode(True))
        self.test_layer.addWidget(self.stop_repeat_bt)

        # --- Watchdog Box ---
        self.watchdog_layout = QHBoxLayout()
        self.watchdog_lb = self.make_label('Watchdog')
        self.watchdog_layout.addWidget(self.watchdog_lb)

        self.watchdog_tb = QTextBrowser()
        self.watchdog_tb.setStyleSheet("background-color: #222; color: #ff0;")
        self.watchdog_tb.setFixedHeight(50)
        self.watchdog_tb.setMaximumWidth(500)
        self.watchdog_layout.addWidget(self.watchdog_tb)

        # --- Memory Tracking Box ---
        self.meminfo_layout = QHBoxLayout()
        self.meminfo_bt = self.make_button('&Memory Check', width=DEFAULT_LABEL_WIDTH, handler=self.memory_check)
        self.meminfo_bt.setFixedHeight(50)
        self.meminfo_layout.addWidget(self.meminfo_bt)

        self.meminfo_tb = QTextBrowser()
        self.meminfo_tb.setStyleSheet("background-color: #222; color: #ff0;")
        self.meminfo_tb.setFixedHeight(50)
        self.meminfo_tb.setMaximumWidth(500)
        self.meminfo_layout.addWidget(self.meminfo_tb)

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

        # --- Shell response ---
        self.shell_resp_layout = QHBoxLayout()

        self.shell_tb = QTextBrowser()
        self.shell_tb.setStyleSheet("background-color: black; color: white;")
        self.shell_tb.setAcceptRichText(True)
        self.shell_tb.setOpenExternalLinks(True)
        self.shell_tb.setMinimumHeight(600)
        self.shell_tb.setMinimumWidth(600)

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

        # --- cmd.exe start ---
        self.cmd_process = QProcess(self)
        self.cmd_process.setProcessChannelMode(QProcess.MergedChannels)
        self.cmd_process.readyReadStandardOutput.connect(self.read_cmd_output)
        self.cmd_process.start("cmd.exe")

        # --- layer options ---
        self.left_layout = QVBoxLayout()
        self.left_layout.addLayout(self.port_layout)
        self.left_layout.addLayout(self.relay_layout)
        self.left_layout.addLayout(self.washer_cont_layout)
        self.left_layout.addLayout(self.disp_cont_layout)
        self.left_layout.addLayout(self.touch_layout)
        self.left_layout.addLayout(self.app_start_layout)
        self.left_layout.addLayout(self.app_quit_layout)
        self.left_layout.addLayout(self.test_layer)
        self.left_layout.addLayout(self.watchdog_layout)
        self.left_layout.addLayout(self.meminfo_layout)
        self.left_layout.addLayout(self.cmd_layout)
        self.left_layout.addWidget(self.cmd_tb)

        self.right_layout = QVBoxLayout()
        self.right_layout.addLayout(self.shell_input_layout)
        self.right_layout.addWidget(self.shell_tb)

        self.top_layout = QHBoxLayout()
        self.top_layout.addLayout(self.left_layout, stretch=0)
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
    def app_start_process(self, idx):
        self.app_start_bts[idx].setStyleSheet('background:#feffcd; color:black;')
        self.shell_cmd(f"app start {DEMO_APPS[idx]}")
        self.app_start_bts[idx].setEnabled(False)
        self.is_app_running[idx] = True

    def app_all_start_process(self):
        for idx in range(4):
            if not self.is_app_running[idx]:
                QTimer.singleShot(idx * 500, lambda idx=idx: self.app_start_process(idx))

    # --- app quit function ---
    def app_quit_process(self, idx, mode):
        self.shell_cmd(f"app quit {DEMO_APPS[idx]}")
        if mode:
            self.app_start_bts[idx].setStyleSheet('background-color: #e0e0e0; color: black;')
            self.app_start_bts[idx].setEnabled(True)
        self.is_app_running[idx] = False

    def app_all_quit_process(self, mode):
        for idx in range(4):
            if self.is_app_running[idx]:
                QTimer.singleShot(idx * 500, lambda idx=idx: self.app_quit_process(idx, mode))

    # --- repeat start function ---
    def start_repeat_mode(self, idx):
        self.repeat_flags[idx] = True
        self.app_start_process(idx)
        self.start_repeat_bts[idx].setText(f'App #{idx+1} ({self.repeat_counts[idx]+1})')
        self.is_app_running[idx] = True
        self.start_repeat_bts[idx].setStyleSheet('background:#feffcd; color:black;')
        self.start_repeat_bts[idx].setEnabled(False)
        self.meminfo_tb.clear()
        self.watchdog_tb.clear()
        self.watchdog_lb.setStyleSheet('color:black; background:#e0e0e0;')
        self.repeat_timers[idx].start(DEFAULT_REPEAT_INTERVAL, self)

    # --- repeat stop function ---
    def stop_repeat_mode(self, mode):
        for idx in range(4):
            if self.repeat_flags[idx]:
                self.repeat_flags[idx] = False
                self.repeat_timers[idx].stop()
                self.repeat_counts[idx] = 0
                if mode:
                    QTimer.singleShot(idx * 500, lambda idx=idx: self.start_repeat_bts[idx].setStyleSheet('background:#e0e0e0; color: black;'))
                    self.start_repeat_bts[idx].setEnabled(True)
                self.start_repeat_bts[idx].setText(f'App #{idx+1}')
        self.app_all_quit_process(mode)
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

    # --- Layer options ---
    def set_width_combobox(self, size):
        combos = [self.port_cb, self.lcp_port_cb, self.osp_port_cb]
        for combo in combos:
            combo.setMaximumWidth(size)

    def set_buttons(self, set_enabled=True):
        buttons = (
            [self.disconnect_bt, self.washer_power_bt, self.washer_start_bt, self.disp_on_bt, self.disp_off_bt, self.disp_bl_bt,
            self.app_all_start_bt, self.app_all_quit_bt, self.r_touch_bt, self.l_touch_bt, self.auto_touch_bt, self.auto_touch_stop_bt,
            self.stop_repeat_bt, self.meminfo_bt]
            + self.app_start_bts
            + self.app_quit_bts
            + self.start_repeat_bts
        )
        for button in buttons:
            button.setEnabled(set_enabled)
            if set_enabled:
                button.setStyleSheet('background-color: #e0e0e0; color: black;')
            else:
                button.setStyleSheet('background-color: #b0b0b0;')

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
        self.stop_repeat_mode(False)
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
        for app in DEMO_APPS:
            self.shell_cmd(f"pkg uninstall {app}")

    def left_touch(self):
        self.shell_cmd("input tab 77 318")

    def right_touch(self):
        self.shell_cmd("input tab 923 318")

    def auto_touch_process(self):
        self.shell_tb.append("Auto touch start.")
        self.auto_touch_bt.setStyleSheet('background:#feffcd; color:black;')
        self.auto_touch_bt.setEnabled(False)
        self.touch_timer.start(DEFAULT_TOUCH_INTERVAL, self)

    def auto_touch_stop_process(self):
        self.shell_tb.append("Auto touch stopped.")
        self.auto_touch_bt.setStyleSheet('background:#e0e0e0; color: black;')
        self.auto_touch_bt.setEnabled(True)
        self.touch_timer.stop()

    def touch_timer_callback(self):
        self.shell_cmd("input tab 923 318")

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
                    self.start_repeat_bts[idx].setText(f'App #{idx+1} ({self.repeat_counts[idx]+1})')
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
