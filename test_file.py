import pytest
import nbformat
from hashlib import md5
import pandas as pd



class TestJupyter:
    @pytest.fixture(autouse=True)
    def get_notebook(self):
        with open('Users/kkartikgoel/Superhero.ipynb') as f:
            nb = nbformat.read(f, as_version=4)
        self.nb = nb


    def test_thor(self):
        output = self.nb.cells[6].outputs[0].data['text/plain']
        output = output.encode('utf-8')
        assert md5(output).hexdigest()== 'afbc48a7ca8d716f9efa7cc993316668'

    def test_male(self):
        output = self.nb.cells[8].outputs[0].data['text/plain']
        output = output.encode('utf-8')
        assert md5(output).hexdigest()== '451d13a5be2581a451c2284dcecddd4e'

    def test_mt(self):
        output = self.nb.cells[10].outputs[0].data['text/plain']
        output = output.encode('utf-8')
        assert md5(output).hexdigest()== 'c7c8f9b16ebc7282887baeca67236cbe'

    def test_tm(self):
        output = self.nb.cells[12].outputs[0].data['text/plain']
        output = output.encode('utf-8')
        assert md5(output).hexdigest()== '676813538852c7111802c95f5ca99e41'

