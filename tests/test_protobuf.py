import sys
import os

import pytest

sys.path = [os.path.join(os.path.dirname(__file__), "..")] + sys.path
from physt.io.protobuf import write, write_many, read, read_many
from physt.examples import normal_h1, normal_h2


class TestProtobuf(object):
    # End-to-end test
    def test_h1(self):
        H = normal_h1()
        message = write(H)
        H_ = read(message)
        assert H_ == H

    def test_h2(self):
        H = normal_h2()
        message = write(H)
        H_ = read(message)
        assert H_ == H

    def test_collection(self):
        collection = {
            "h1": normal_h1(),
            "h2": normal_h2()
        }
        message = write_many(collection)
        collection2 = read_many(message)
        assert collection == collection2

    def test_wrong_message_type(self):
        H = normal_h1()
        message = write(H)
        with pytest.raises(AttributeError):
            collection2 = read_many(message)

    # TODO: Add more tests


if __name__ == "__main__":
    pytest.main(__file__)
