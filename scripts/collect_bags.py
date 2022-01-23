import argparse
import curses
import os
import subprocess
import time


# TOPICS_TO_RECORD = [
#     '/tf_static',
#     '/tf',
#     '/zedm/zed_node/left_raw/camera_info',
#     '/zedm/zed_node/left_raw/image_raw_color',
#     '/zedm/zed_node/right_raw/camera_info',
#     '/zedm/zed_node/right_raw/image_raw_color',
#     '/joint_states'
# ]

TOPICS_TO_RECORD = [
    '/tf_static',
    '/tf',
    # '/camera/color/image_raw',
    '/camera/color/image_raw/downsample',
    '/camera/color/camera_info',
    '/camera/depth/camera_info'
    '/panda/joint_states'
]


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', '-o', type=str, default="~/data/bags")
    return parser.parse_args()

WAITING = "Waiting for command."
STARTING = "Starting to record bag."
RECORDING = "Recording bag..."

class Program:
    def __init__(self, screen, flags):
        self.screen = screen
        self.flags = flags
        self._stdout = []
        self.status_line = WAITING
        self._inventory()
        self._refresh_screen()

    def _inventory(self):
        files = sorted(os.listdir(self.flags.out))
        self.current_file = 0
        self._recorded_bags = []
        for f in files:
            filepath = os.path.join(self.flags.out, f)
            if '.bag' in f:
                self._recorded_bags.append(filepath)
                self.current_file += 1

    def _refresh_screen(self):
        self.screen.clear()
        self.screen.addstr(0, 0, self.status_line)
        for i, filepath in enumerate(self._recorded_bags):
            bagname = os.path.basename(filepath)
            self.screen.addstr(i + 2, 0, bagname)

        (height, width) = self.screen.getmaxyx()
        for i, line in enumerate(self._stdout[-20:]):
            self.screen.addstr(height // 2 + i, 0, line)
        self.screen.refresh()

    def _add_bag(self, filepath):
        self._recorded_bags.append(filepath)
        self.current_file += 1

    def _read_stdout(self, process):
        text = process.stdout.decode('utf-8')
        for line in text.split('\n'):
            self._stdout.append(line)

    def _record_bag(self):
        self.status_line = STARTING
        self._refresh_screen()
        time.sleep(5)
        filename = '{:03}.bag'.format(self.current_file)
        filepath = os.path.join(self.flags.out, filename)
        self.status_line = RECORDING
        self._refresh_screen()
        try:
            process = subprocess.run(['rosbag', 'record', '--buffsize=0', '--chunksize=524288', '--output-name', filepath, '--duration', '15'] + TOPICS_TO_RECORD,
                    stdout=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as e:
            print(e)
            exit()

        self._read_stdout(process)
        self._add_bag(filepath)
        self.status_line = WAITING
        self._refresh_screen()

    def run(self):
        while True:
            keypress = self.screen.getkey()
            if keypress == 'q':
                curses.endwin()
                return
            elif keypress == '\n':
                self._record_bag()

def main(screen):
    curses.noecho()
    flags = read_args()
    flags.out = os.path.expanduser(flags.out)

    os.makedirs(flags.out, exist_ok=True)

    program = Program(screen, flags)
    program.run()


if __name__ == "__main__":
    curses.wrapper(main)


