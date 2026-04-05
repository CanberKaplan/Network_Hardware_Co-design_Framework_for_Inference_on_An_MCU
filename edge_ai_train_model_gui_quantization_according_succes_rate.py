import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QPushButton, QTextEdit, QCheckBox, QGroupBox, QFrame, QComboBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# ==========================================
# 1. MODEL DEFINITION
# ==========================================
class ScalableDiagramCNN(nn.Module):
    def __init__(self, channels, input_h=28, input_w=28, num_classes=10, num_layers=2):
        super(ScalableDiagramCNN, self).__init__()
        self.channels = channels
        self.num_layers = num_layers
        layers = []
        in_ch = 1
        
        for i in range(num_layers):
            layers.append(nn.Conv2d(in_ch, channels, kernel_size=3, stride=1, padding=0))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_ch = channels
            
        self.features = nn.Sequential(*layers)
        
        curr_h, curr_w = input_h, input_w
        for _ in range(num_layers):
            curr_h = (curr_h - 2) // 2
            curr_w = (curr_w - 2) // 2
            
        if curr_h <= 0 or curr_w <= 0:
            raise ValueError(f"Input shape {input_h}x{input_w} is too small for {num_layers} layers.")
            
        self.fc = nn.Linear(channels * curr_h * curr_w, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# ==========================================
# 2. QUANTIZED C-HEADER EXPORT (INT8 / INT16)
# ==========================================
def write_c_array_quantized(file_obj, name, tensor_data, is_int=False, quant_mode="INT16"):
    data_np = tensor_data.detach().cpu().numpy().flatten()
    
    if is_int:
        data_np = data_np.astype(np.int32)
        c_type = "int32_t"
        format_str = "{}"
    else:
        if quant_mode == "INT8":
            max_val = np.max(np.abs(data_np))
            scale = 127.0 / max_val if max_val > 0 else 1.0
            data_np = np.round(data_np * scale).astype(np.int8)
            c_type = "int8_t"
        else:
            max_val = np.max(np.abs(data_np))
            scale = 32767.0 / max_val if max_val > 0 else 1.0
            data_np = np.round(data_np * scale).astype(np.int16)
            c_type = "int16_t"
            
        format_str = "{}"
        file_obj.write(f"// Quantization Scale: {scale:.6f}f\n")

    size = len(data_np)
    shape_str = "x".join(map(str, tensor_data.shape))
    file_obj.write(f"// Original Shape: [{shape_str}]\n")
    file_obj.write(f"const {c_type} {name}[{size}] = {{\n")
    
    values = [format_str.format(val) for val in data_np]
    for i in range(0, size, 10):
        line = ", ".join(values[i:i+10])
        if i + 10 < size:
            line += ","
        file_obj.write(f"    {line}\n")
    file_obj.write("};\n\n")

# ==========================================
# 3. ARCHITECTURE SEARCH & TRAINING THREAD
# ==========================================
class TrainingThread(QThread):
    update_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    optimal_arch_found_signal = pyqtSignal(int) 

    def __init__(self, max_channels, layers, quant_mode, input_h, input_w, num_classes, target_conf):
        super().__init__()
        self.max_channels = max_channels
        self.layers = layers
        self.quant_mode = quant_mode
        self.input_h = input_h
        self.input_w = input_w
        self.num_classes = num_classes
        self.target_conf = target_conf
        self.max_epochs_per_arch = 4

    def run(self):
        try:
            self.update_signal.emit("\n<b>[1/4] Preparing Dataset...</b>")
            transform = transforms.Compose([
                transforms.Resize((self.input_h, self.input_w)),
                transforms.ToTensor(), 
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            os.makedirs('./data', exist_ok=True)
            os.makedirs('./export', exist_ok=True) 

            mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            
            subset_indices = torch.randperm(len(mnist_train))[:10000]
            train_subset = torch.utils.data.Subset(mnist_train, subset_indices)
            
            train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
            test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=1000, shuffle=False)

            self.update_signal.emit(f"<b>[2/4] Architecture Search (Target Confident Acc: {self.target_conf}%)</b>")
            
            optimal_model = None
            optimal_ch = self.max_channels
            target_met = False

            # --- TRAINING HELPER FUNCTION ---
            def test_architecture(ch):
                model = ScalableDiagramCNN(ch, self.input_h, self.input_w, self.num_classes, self.layers)
                
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.005) 
                
                final_conf_acc = 0
                for epoch in range(self.max_epochs_per_arch):
                    model.train()
                    for images, labels in train_loader:
                        optimizer.zero_grad()
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                    
                    model.eval()
                    correct = 0
                    confident_and_correct = 0
                    total = 0
                    
                    with torch.no_grad():
                        for images, labels in test_loader:
                            outputs = model(images)
                            probabilities = torch.softmax(outputs.data, dim=1)
                            max_probs, predicted = torch.max(probabilities, 1)
                            
                            total += labels.size(0)
                            correct_mask = (predicted == labels)
                            correct += correct_mask.sum().item()
                            
                            conf_mask = (max_probs >= (self.target_conf / 100.0))
                            confident_and_correct += (correct_mask & conf_mask).sum().item()
                    
                    accuracy = 100 * correct / total
                    final_conf_acc = 100 * confident_and_correct / total
                    
                    self.update_signal.emit(f"  -> Epoch {epoch+1}/{self.max_epochs_per_arch} | Acc: {accuracy:.1f}% | <b>Conf Acc: {final_conf_acc:.1f}%</b>")
                    
                    if final_conf_acc >= self.target_conf:
                        break 
                        
                return model, final_conf_acc

            # PHASE 1: FAST SEARCH (10-Channel Increments)
            self.update_signal.emit("<br><span style='color:#3498db;'><b>--- PHASE 1: Fast Search (10-Channel Increments) ---</b></span>")
            current_ch = min(10, self.max_channels)
            
            while True:
                self.update_signal.emit(f"<br><i>Testing Memory Profile with {current_ch} Channels...</i>")
                model, conf_acc = test_architecture(current_ch)
                
                if conf_acc >= self.target_conf:
                    self.update_signal.emit(f"<span style='color:#f1c40f;'><b>★ Target met at {current_ch} channels! Moving to Phase 2... ★</b></span>")
                    optimal_model = model
                    optimal_ch = current_ch
                    target_met = True
                    break
                
                if current_ch == self.max_channels:
                    optimal_model = model
                    break
                    
                current_ch += 10
                if current_ch > self.max_channels:
                    current_ch = self.max_channels

            # PHASE 2: FINE-TUNING (Step-by-step Reduction)
            if target_met and optimal_ch > 1:
                self.update_signal.emit("<br><span style='color:#e67e22;'><b>--- PHASE 2: Fine-Tuning (Step-by-step Reduction) ---</b></span>")
                lower_bound = max(1, optimal_ch - 9) 
                
                for fine_ch in range(optimal_ch - 1, lower_bound - 1, -1):
                    self.update_signal.emit(f"<br><i>Testing Fine-tune: {fine_ch} Channels...</i>")
                    fine_model, fine_conf_acc = test_architecture(fine_ch)
                    
                    if fine_conf_acc >= self.target_conf:
                        self.update_signal.emit(f"<span style='color:#2ecc71;'><b>  -> {fine_ch} channels also reached the target. Continuing to scale down...</b></span>")
                        optimal_model = fine_model
                        optimal_ch = fine_ch
                    else:
                        self.update_signal.emit(f"<span style='color:#e74c3c;'><b>  -> Target failed at {fine_ch} channels. Minimum optimal model: {optimal_ch} channels.</b></span>")
                        break

            if not target_met:
                self.update_signal.emit(f"<br><span style='color:#e74c3c;'><b>⚠️ Could not hit target Confident Acc within hardware limits. Exporting best attempt.</b></span>")

            # EXPORT C FILES AND FINISH
            self.optimal_arch_found_signal.emit(optimal_ch)

            self.update_signal.emit("<br><b>[3/4] Saving Final Model...</b>")
            model_file = f'./export/edge_cnn_{self.quant_mode.lower()}_{optimal_ch}ch.pth'
            torch.save(optimal_model.state_dict(), model_file)

            self.update_signal.emit(f"<b>[4/4] Generating {self.quant_mode} C Header Files...</b>")
            header_suffix = self.quant_mode.lower()
            macro_prefix = self.quant_mode.upper()
            
            with open(f'./export/model_weights_{header_suffix}.h', 'w') as f:
                f.write(f"#ifndef MODEL_WEIGHTS_{macro_prefix}_H\n#define MODEL_WEIGHTS_{macro_prefix}_H\n\n")
                f.write("#include <stdint.h>\n\n")
                
                conv_idx = 1
                for layer in optimal_model.features:
                    if isinstance(layer, nn.Conv2d):
                        write_c_array_quantized(f, f"conv{conv_idx}_weights", layer.weight, quant_mode=self.quant_mode)
                        write_c_array_quantized(f, f"conv{conv_idx}_bias", layer.bias, quant_mode=self.quant_mode)
                        conv_idx += 1
                        
                write_c_array_quantized(f, "fc_weights", optimal_model.fc.weight, quant_mode=self.quant_mode)
                write_c_array_quantized(f, "fc_bias", optimal_model.fc.bias, quant_mode=self.quant_mode)
                f.write("#endif\n")

            found_digits = {}
            with open(f'./export/test_samples_{header_suffix}.h', 'w') as f:
                f.write(f"#ifndef TEST_SAMPLES_{macro_prefix}_H\n#define TEST_SAMPLES_{macro_prefix}_H\n\n")
                f.write("#include <stdint.h>\n\n")
                optimal_model.eval()
                with torch.no_grad():
                    for idx in range(len(mnist_test)):
                        if len(found_digits) == self.num_classes: break 
                        img, label = mnist_test[idx]
                        if label >= self.num_classes: continue 
                        
                        output = optimal_model(img.unsqueeze(0))
                        predicted = torch.argmax(output).item()
                        if predicted == label and label not in found_digits:
                            write_c_array_quantized(f, f"test_img_class_{label}", img, quant_mode=self.quant_mode)
                            write_c_array_quantized(f, f"test_lbl_class_{label}", torch.tensor([label]), is_int=True)
                            found_digits[label] = True
                f.write("#endif\n")

            self.update_signal.emit(f"<br><span style='color:#2ecc71;'><b>🚀 SUCCESS! Optimized {self.quant_mode} Model and C files are ready.</b></span>")
            self.finished_signal.emit()

        except Exception as e:
            self.update_signal.emit(f"<br><span style='color:#e74c3c;'><b>❌ ERROR: {str(e)}</b></span>")
            self.finished_signal.emit()


# ==========================================
# 4. MAIN INTERFACE (GUI)
# ==========================================
class EdgeAiGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Edge AI: NAS Target Accuracy & SRAM Manager")
        self.setMinimumSize(1200, 850)
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_panel = QVBoxLayout()
        input_group = QGroupBox("Hardware Limits & Network Parameters")
        input_grid = QVBoxLayout()
        
        quant_layout = QHBoxLayout()
        quant_layout.addWidget(QLabel("Quantization:"))
        self.quant_combo = QComboBox()
        self.quant_combo.addItems(["INT16", "INT8"])
        quant_layout.addWidget(self.quant_combo)
        input_grid.addLayout(quant_layout)

        self.target_conf_input = self.create_input("Target Confident Accuracy (%):", "90")
        self.input_shape_input = self.create_input("Input Shape (H,W):", "28,28")
        self.num_classes_input = self.create_input("Number of Classes:", "10")
        self.layers_input = self.create_input("Number of Layers (1-2):", "2")
        self.flash_input = self.create_input("Max Flash (KB):", "128")
        self.conv_sram_input = self.create_input("Conv SRAM (KB):", "32")
        self.fc_sram_input = self.create_input("Dense SRAM (KB):", "8")
        self.arena_check = QCheckBox("Ultimate Arena (Input Excluded)")
        self.arena_check.setChecked(True)
        
        for inp in [self.target_conf_input, self.input_shape_input, self.num_classes_input, self.layers_input, self.flash_input, self.conv_sram_input, self.fc_sram_input]:
            input_grid.addLayout(inp[0])
        input_grid.addWidget(self.arena_check)
        input_group.setLayout(input_grid)
        
        self.calc_btn = QPushButton("1. CALCULATE HARDWARE CEILING")
        self.calc_btn.setFixedHeight(50)
        self.calc_btn.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold;")
        self.calc_btn.clicked.connect(self.run_logic)
        
        self.train_btn = QPushButton("2. START ARCHITECTURE SEARCH")
        self.train_btn.setFixedHeight(50)
        self.train_btn.setEnabled(False)
        self.train_btn.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold;")
        self.train_btn.clicked.connect(self.start_training)

        left_panel.addWidget(input_group)
        left_panel.addWidget(self.calc_btn)
        left_panel.addWidget(self.train_btn)
        left_panel.addStretch()

        right_panel = QVBoxLayout()
        self.log_output = QTextEdit()
        self.log_output.setFixedHeight(220)
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("background-color: #121212; color: #00FF00; font-family: Consolas;")
        
        self.diagram_container = QGroupBox("Architecture and Step-Based Memory Consumption")
        self.diagram_layout = QVBoxLayout()
        self.diagram_container.setLayout(self.diagram_layout)
        
        self.arena_status_label = QLabel("Status: Waiting for Calculation")
        self.arena_status_label.setAlignment(Qt.AlignCenter)

        right_panel.addWidget(QLabel("<b>Detailed Analysis & Training Log:</b>"))
        right_panel.addWidget(self.log_output)
        right_panel.addWidget(self.diagram_container)
        right_panel.addWidget(self.arena_status_label)

        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 3)

    def create_input(self, label_text, default_val):
        layout = QHBoxLayout()
        edit = QLineEdit(default_val)
        layout.addWidget(QLabel(label_text))
        layout.addWidget(edit)
        return layout, edit

    def calculate_metrics(self, channels, layers, use_arena, quant_mode, input_h, input_w, num_classes):
        dtype_size = 2 if quant_mode == "INT16" else 1
        total_flash = 0
        in_ch = 1 
        curr_h, curr_w = input_h, input_w
        
        input_ram_fixed = (1 * input_h * input_w * dtype_size)
        max_internal_buffer = 0
        cumulative_internal_buffer = 0
        
        for i in range(layers):
            p = (3 * 3 * in_ch * channels) + channels
            total_flash += p * dtype_size
            
            mem_conv_out = channels * (curr_h - 2) * (curr_w - 2) * dtype_size
            mem_pool_out = channels * ((curr_h - 2)//2) * ((curr_w - 2)//2) * dtype_size
            
            if use_arena:
                max_internal_buffer = max(max_internal_buffer, mem_conv_out, mem_pool_out)
            else:
                cumulative_internal_buffer += (mem_conv_out + mem_pool_out)
                
            curr_h = (curr_h - 2) // 2
            curr_w = (curr_w - 2) // 2
            in_ch = channels

        total_conv_sram = input_ram_fixed + (max_internal_buffer if use_arena else cumulative_internal_buffer)
        
        fc_in = channels * curr_h * curr_w
        fc_p = (fc_in * num_classes) + num_classes
        
        # Dense SRAM Explicit Calculations 
        flatten_sram_bytes = fc_in * dtype_size
        dense_weights_sram_bytes = fc_p * dtype_size
        output_sram_bytes = num_classes * dtype_size
        
        fc_sram = flatten_sram_bytes + dense_weights_sram_bytes + output_sram_bytes
        
        return total_flash, total_conv_sram, fc_sram, fc_in

    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
            elif item.layout(): self.clear_layout(item.layout())

    def create_box(self, title, subtitle, f_kb, s_kb, color="#f1c40f", sram_label="SRAM"):
        box = QFrame()
        box.setStyleSheet(f"background-color: #222; border: 2px solid {color}; border-radius: 8px;")
        box.setFixedWidth(190); box.setFixedHeight(120)  # slightly heightened to fit new label comfortably
        l = QVBoxLayout(box)
        t = QLabel(title); t.setAlignment(Qt.AlignCenter); t.setStyleSheet("font-weight: bold; color:white; border:none;")
        st = QLabel(subtitle); st.setAlignment(Qt.AlignCenter); st.setStyleSheet("font-size: 10px; color:#bdc3c7; border:none;")
        
        # Note the sram_label injection
        m = QLabel(f"Flash: {f_kb:.2f}K\n{sram_label}: {s_kb:.2f}K")
        m.setAlignment(Qt.AlignCenter); m.setStyleSheet("color:#2ecc71; border:none; font-size:9px;")
        
        l.addWidget(t); l.addWidget(st); l.addWidget(m)
        return box

    def update_diagram(self, channels, layers, use_arena, quant_mode, input_h, input_w, num_classes):
        self.clear_layout(self.diagram_layout)
        
        # Create grouped boxes for distinct visualizations
        conv_group = QGroupBox("Convolutional Memory Domain (Conv SRAM)")
        conv_group.setStyleSheet("QGroupBox { border: 1px solid #7f8c8d; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #f1c40f; }")
        cnn_row = QHBoxLayout()
        cnn_row.setAlignment(Qt.AlignLeft)
        conv_group.setLayout(cnn_row)

        dense_group = QGroupBox("Dense Memory Domain (Dense SRAM)")
        dense_group.setStyleSheet("QGroupBox { border: 1px solid #7f8c8d; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #3498db; }")
        dense_row = QHBoxLayout()
        dense_row.setAlignment(Qt.AlignLeft)
        dense_group.setLayout(dense_row)

        dtype_size = 2 if quant_mode == "INT16" else 1
        
        input_sram = (input_h * input_w * dtype_size) / 1024 
        cnn_row.addWidget(self.create_box(f"{input_h}x{input_w}", "INPUT SIGNAL", 0, input_sram, "#95a5a6", "Conv SRAM"))
        
        curr_h, curr_w = input_h, input_w
        in_ch = 1
        for i in range(layers):
            cnn_row.addWidget(QLabel("➔"))
            curr_h -= 2
            curr_w -= 2
            f_conv = ((3*3*in_ch*channels)+channels)*dtype_size/1024
            s_conv = (channels * curr_h * curr_w * dtype_size)/1024
            cnn_row.addWidget(self.create_box(f"{channels}x{curr_h}x{curr_w}", f"CONV {i+1}", f_conv, s_conv, "#f1c40f", "Conv SRAM"))
            
            cnn_row.addWidget(QLabel("➔"))
            curr_h //= 2
            curr_w //= 2
            s_pool = (channels * curr_h * curr_w * dtype_size)/1024
            cnn_row.addWidget(self.create_box(f"{channels}x{curr_h}x{curr_w}", f"POOL {i+1}", 0, s_pool, "#e67e22", "Conv SRAM"))
            in_ch = channels
        
        fc_in = channels * curr_h * curr_w
        
        # Dense Memory SRAM Calculations
        flatten_sram_kb = (fc_in * dtype_size) / 1024
        dense_w_mem_kb = (fc_in * num_classes + num_classes) * dtype_size / 1024
        output_sram_kb = (num_classes * dtype_size) / 1024
        
        dense_row.addWidget(self.create_box(f"{fc_in}x1", "FLATTEN", 0, flatten_sram_kb, "#3498db", "Dense SRAM"))
        dense_row.addWidget(QLabel(" X "))
        
        dense_row.addWidget(self.create_box(f"{num_classes}x{fc_in}", "DENSE W", 0, dense_w_mem_kb, "#9b59b6", "Dense SRAM"))
        
        dense_row.addWidget(QLabel(" = "))
        dense_row.addWidget(self.create_box(f"{num_classes}x1", "OUTPUT", 0, output_sram_kb, "#2ecc71", "Dense SRAM"))
        
        # Add the grouped boxes to the main diagram layout
        self.diagram_layout.addWidget(conv_group)
        self.diagram_layout.addWidget(dense_group)

    def run_logic(self):
        try:
            shape_str = self.input_shape_input[1].text().split(',')
            self.input_h = int(shape_str[0].strip())
            self.input_w = int(shape_str[1].strip()) if len(shape_str) > 1 else self.input_h
            
            self.classes_val = int(self.num_classes_input[1].text())
            f_lim = float(self.flash_input[1].text()) * 1024
            c_lim = float(self.conv_sram_input[1].text()) * 1024
            d_lim = float(self.fc_sram_input[1].text()) * 1024
            self.layers_val = int(self.layers_input[1].text())
            self.use_arena = self.arena_check.isChecked()
            self.quant_mode = self.quant_combo.currentText()

            self.max_ch = 1
            while True:
                f, sc, sd, fi = self.calculate_metrics(self.max_ch + 1, self.layers_val, self.use_arena, self.quant_mode, self.input_h, self.input_w, self.classes_val)
                if f > f_lim or sc > c_lim or sd > d_lim or self.max_ch >= 128:
                    break
                self.max_ch += 1
            
            res_f, res_sc, res_sd, res_fi = self.calculate_metrics(self.max_ch, self.layers_val, self.use_arena, self.quant_mode, self.input_h, self.input_w, self.classes_val)
            self.update_diagram(self.max_ch, self.layers_val, self.use_arena, self.quant_mode, self.input_h, self.input_w, self.classes_val)
            self.diagram_container.setTitle(f"MAXIMUM Hardware Limits ({self.quant_mode} - KB)")
            
            self.log_output.clear()
            self.log_output.append(f"<b>Maximum Hardware Profile Calculated (Ceiling: {self.max_ch} Channels)</b>")
            self.log_output.append(f"If starting search, we will look for the smallest model between 1 and {self.max_ch} channels.")
            self.arena_status_label.setText(f"Arena Active ({self.quant_mode} Mode)" if self.use_arena else "Arena Disabled (Total SRAM Check)")
            self.train_btn.setEnabled(True)
        except ValueError as ve:
            self.log_output.append(f"<span style='color:#e74c3c;'>Error: {ve}</span>")
        except Exception as e:
            self.log_output.append(f"<span style='color:#e74c3c;'>Error parsing inputs: Check comma formatting (e.g. 28,28)</span>")

    def handle_optimal_architecture(self, optimal_ch):
        res_f, res_sc, res_sd, res_fi = self.calculate_metrics(optimal_ch, self.layers_val, self.use_arena, self.quant_mode, self.input_h, self.input_w, self.classes_val)
        self.update_diagram(optimal_ch, self.layers_val, self.use_arena, self.quant_mode, self.input_h, self.input_w, self.classes_val)
        self.diagram_container.setTitle(f"OPTIMIZED Architecture Found ({self.quant_mode} - KB)")
        self.log_output.append(f"<br><b>Final Footprint ({optimal_ch} Channels):</b>")
        self.log_output.append(f"Flash Usage: {res_f/1024:.2f} KB")
        self.log_output.append(f"Conv SRAM Usage: {res_sc/1024:.2f} KB")
        self.log_output.append(f"Dense SRAM Usage: {res_sd/1024:.2f} KB")  # Ensure Dense SRAM is logged

    def start_training(self):
        self.train_btn.setEnabled(False)
        self.calc_btn.setEnabled(False)
        self.quant_mode = self.quant_combo.currentText()
        
        target_conf = float(self.target_conf_input[1].text()) 
        
        self.log_output.clear()
        self.log_output.append("<b>Initiating NAS (Neural Architecture Search) Protocol...</b>")
        
        self.thread = TrainingThread(self.max_ch, self.layers_val, self.quant_mode, self.input_h, self.input_w, self.classes_val, target_conf)
        self.thread.update_signal.connect(self.log_output.append)
        self.thread.optimal_arch_found_signal.connect(self.handle_optimal_architecture)
        self.thread.finished_signal.connect(self.training_finished)
        self.thread.start()

    def training_finished(self):
        self.train_btn.setEnabled(True)
        self.calc_btn.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EdgeAiGui()
    window.show()
    sys.exit(app.exec_())