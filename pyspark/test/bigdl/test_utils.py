
import numpy as np
from unittest import TestCase

class BigDLTestCase(TestCase):

    def assert_allclose(self, a, b, rtol=1e-6, atol=1e-6, msg=None):
        self.assertEqual(a.shape, b.shape, "Shape mismatch: expected %s, got %s." %
                         (a.shape, b.shape))
        if not np.allclose(a, b, rtol=rtol, atol=atol):
            cond = np.logical_or(
                np.abs(a - b) > atol + rtol * np.abs(b), np.isnan(a) != np.isnan(b))
            if a.ndim:
                x = a[np.where(cond)]
                y = b[np.where(cond)]
                print("not close where = ", np.where(cond))
            else:
                # np.where is broken for scalars
                x, y = a, b
            print("not close lhs = ", x)
            print("not close rhs = ", y)
            print("not close dif = ", np.abs(x - y))
            print("not close tol = ", atol + rtol * np.abs(y))
            print("dtype = %s, shape = %s" % (a.dtype, a.shape))
            np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=msg)