from pathlib import Path
import model_num as mn
import os

class DataLogger:
    _file_handle:Path = None
    _video_directory:Path = None

    def set_file_handle(self, filepath:str):
        self._file_handle = Path(filepath)
        if self._file_handle.exists():
            os.remove(self._file_handle)
        self._file_handle.touch()
        assert filepath.endswith('md')
    
    def get_file_handle(self):
        return self._file_handle

    def write(self, string:str)->bool:
        if self._file_handle is None:
            return False
        if not string.endswith('\n'):
            string += '\n'
        with open(self._file_handle, 'a') as f:
            f.write(string)
        return True
    
    def blank(self)->bool:
        return self.write("")

    def write_top_header(self):
        self.write("# Data")
        self.blank()

    def write_overview(self, **kwargs):
        self.write("| Model Parameter | Value |")
        self.write("| :-------------: | :---: |")
        for p in kwargs:
            self.write("| " + str(p) + " | " + str(kwargs[p]) + " |")
        self.blank()

    def write_for_parameter(self, param:str, d):
        '''
        @Params:
            param:str
                the name of the parameter
            d:list
                a list of 3-tuples that represent the value of the parameter, its wavelength, and its periodicity
        '''
        pass

if __name__ == "__main__":
    # test Datalogger
    dl = DataLogger()
    dl.set_file_handle('./test_file.md')
    dl.write_top_header()
    dl.write_overview(param1=0.1, param2=0.5, paramA = "RK3")
