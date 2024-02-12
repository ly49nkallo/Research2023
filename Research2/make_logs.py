from pathlib import Path
import model_num as mn
import os

class DataLogger:
    _file_handle:Path = None
    _video_directory:Path = None
    
    def set_file_handle(self, filepath:str):
        self._file_handle = Path(filepath)
        self.reset_file()
        assert filepath.endswith('md')
    
    def get_file_handle(self):
        return self._file_handle

    def reset_file(self):
        if self._file_handle is None:
            raise Exception
        if self._file_handle.exists():
            os.remove(self._file_handle)
        self._file_handle.touch() 
        
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
        if len(kwargs) == 0: return
        self.write("| Model Parameter | Value |")
        self.write("| :-------------: | :---: |")
        for p in kwargs:
            self.write("| " + str(p) + " | " + str(kwargs[p]) + " |")
        self.blank()
    
    def write_preamble(self, **kwargs):
        self.write_top_header()
        self.write_overview(**kwargs)

    def write_for_parameter(self, param:str, values_arr:list, wavelength_arr:list, is_periodic_arr:list, time_started, elapsed_time, write_header=True):
        '''
        @Params:
            param:str
                the name of the parameter
        '''
        d = list(zip(values_arr, wavelength_arr, is_periodic_arr))
        if write_header:
            self.write(f'## Parameter \"{param}\"')
            self.blank()
        self.write(f'Time started: {time_started}')
        self.blank()
        self.write("| Value | Wavelength | Periodic? |") 
        self.write("| :---: | :---: | :---: |")
        for v, wl, p in d:
            self.write(f'| {v} | {wl} | {str(p)} |')
        self.blank()
        self.write(f'Elapsed time: {elapsed_time}')
        
        pass

if __name__ == "__main__":
    # test Datalogger
    dl = DataLogger()
    dl.set_file_handle('./test_log.md')
    dl.write_top_header()
    dl.write_overview(param1=0.1, param2=0.5, paramA = "RK3")
    dl.write_for_parameter("Xhb", [0.11], [45], [True], 0, 100.0011)
