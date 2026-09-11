import sys
sys.path.append('/workdir/PorepyPlastic')
import tensor

import numpy as np

testTensor = tensor.Tensor(np.array([[1,2,3],[2,3,4],[3,4,5]]), copy=True)

assert testTensor.is_symmetric == True
assert testTensor.tensor_order == 2

testTensor = tensor.Tensor(np.array([[1,2,3],[2,3,4]]), copy=True)

assert testTensor.tensor_order == 2
#assert testTensor.is_symmetric == False

testTensor = tensor.Tensor(np.array([[[1,2,3],
                                     [2,3,4],
                                     [3,4,5]],
                                     [[3,4,5],
                                     [4,5,6],
                                     [5,6,7]],
                                     [[5,6,7],
                                      [6,7,8],
                                      [7,8,9]]]))

assert testTensor.tensor_order == 3
assert testTensor.is_symmetric == False

testsymmetricSecondOrderTensor = tensor.symmetricSecondOrderTensor(np.array([[1,2,3],[2,3,4],[3,4,5]]),copy=True)

assert testsymmetricSecondOrderTensor.is_symmetric == True
#testsymmetricSecondOrderTensor = tensor.symmetricSecondOrderTensor(np.array([1,2,3]))

testsymmetricSecondOrderTensor.create_mandel_representation()
print(testsymmetricSecondOrderTensor.mandel_vec)

# test arithmetric operations
testsymmetricSecondOrderTensor1 = tensor.symmetricSecondOrderTensor(np.array([[1,2,3],[2,3,4],[3,4,5]]))
testsymmetricSecondOrderTensor2 = tensor.symmetricSecondOrderTensor(np.array([[1,2,3],[2,3,4],[3,4,5]]))

testsymmetricSecondOrderTensor3 = testsymmetricSecondOrderTensor1 + testsymmetricSecondOrderTensor2

testsymmetricSecondOrderTensor3 = testsymmetricSecondOrderTensor1 + tensor.symmetricSecondOrderTensor.identity(3)

testsymmetricSecondOrderTensor3 = testsymmetricSecondOrderTensor1 @ testsymmetricSecondOrderTensor2

print(testsymmetricSecondOrderTensor3.shape)
print(testsymmetricSecondOrderTensor1._data)
print(testsymmetricSecondOrderTensor2._data)
print(testsymmetricSecondOrderTensor3._data)

testTensor1 = tensor.Tensor(np.array([[1,2,3]]))
testTensor2 = tensor.Tensor(np.array([[1],[2],[3]]))
testTensor3 = testTensor1 @ testTensor2

print(testTensor3.shape)
print(testTensor1._data)
print(testTensor2._data)
print(testTensor3._data)

# test arithmetric operations
teststress1 = tensor.stress(np.array([[1,2,3],[2,3,4],[3,4,5]]))
teststress2 = tensor.stress(np.array([[1,2,3],[2,3,4],[3,4,5]]))

teststress3 = teststress1 + teststress2

teststress3 = teststress1 + tensor.stress.identity(3)

teststress3 = teststress1 @ teststress2

print(teststress3.shape)
print(teststress1._data)
print(teststress2._data)
print(teststress3._data)

testTensor1 = tensor.Tensor(np.array([[1,2,3]]))
testTensor2 = tensor.Tensor(np.array([[1],[2],[3]]))
testTensor3 = testTensor1 @ testTensor2

print(testTensor3.shape)
print(testTensor1._data)
print(testTensor2._data)
print(testTensor3._data)

teststress1.create_mandel_representation()
teststress4 = tensor.stress.from_mandel(teststress1.mandel_vec)
print(teststress4._data)

