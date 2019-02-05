import sys
import os

import pytest

sys.path = [os.path.join(os.path.dirname(__file__), "..")] + sys.path

try:
    from physt.io.protobuf import write, write_many, read, read_many
    # PROTOBUF_TEST_ENABLED = os.environ.get("PROTOBUF_TEST_ENABLED", False)
    PROTOBUF_TEST_ENABLED = True
except:
    PROTOBUF_TEST_ENABLED = False

from physt.examples import normal_h1, normal_h2


@pytest.mark.skipif(not PROTOBUF_TEST_ENABLED, reason="Skipping protobuf tests because of an error")
class TestProtobuf:
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
