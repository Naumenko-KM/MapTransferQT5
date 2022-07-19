import sys

from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QComboBox)

# Declare class to create the ComboBox

class ComboExample(QWidget):

    def __init__(self):

        super().__init__()


        # Set the label before the ComboBox

        self.topLabel = QLabel('Select your favorite programming language:', self)


        # Define the combobox with items

        combobox = QComboBox(self)

        combobox.addItem('PHP')

        combobox.addItem('Python')

        combobox.addItem('Perl')

        combobox.addItem('Bash')

        combobox.addItem('Java')


        # Set the label after the ComboBox

        self.bottomLabel = QLabel('', self)

        self.bottomLabel.adjustSize()


        # Define vartical layout box

        v_layout = QVBoxLayout()

        v_layout.addWidget(self.topLabel)

        v_layout.addWidget(combobox)

        v_layout.addWidget(self.bottomLabel)


        # Call the custom method if any item is selected

        combobox.activated[str].connect(self.onSelected)


        # Set the configurations for the window

        self.setContentsMargins(20, 20, 20, 20)

        self.setLayout(v_layout)

        self.move(800, 300)

        self.setWindowTitle('Use of ComboBox')


    # Custom function to read the value of the selected item

    def onSelected(self, txtVal):

        txtVal = "\nYou have selected: " + txtVal

        self.bottomLabel.setText(txtVal)


# Create app object and execute the app

app = QApplication(sys.argv)

combobox = ComboExample()

combobox.show()

app.exec()