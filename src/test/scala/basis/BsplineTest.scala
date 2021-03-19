package fda
package basis

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.abs
import fda.math.linalg.allClose
import org.scalatest.FunSuite

class BsplineTest extends FunSuite {

  test("testPenalty") {
    val U = Vector[Double](0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5)
    val bs = Bspline((0.0, 5.0), 8, 3, U)
    val truth = DenseMatrix(
      (4.0 / 3.0, -1.0, -1.0 / 3.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      (-1.0, 4.0 / 3.0, -1.0 / 6.0, -1.0 / 6.0, 0.0, 0.0, 0.0, 0.0),
      (-1.0 / 3.0, -1.0 / 6.0, 1.0, -1.0 / 3.0, -1.0 / 6.0, 0.0, 0.0, 0.0),
      (0.0, -1.0 / 6.0, -1.0 / 3.0, 1.0, -1.0 / 6.0, -1.0 / 3.0, 0.0, 0.0),
      (0.0, 0.0, -1.0 / 6.0, -1.0 / 6.0, 4.0 / 3.0, -1.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, -1.0 / 3.0, -1.0, 8.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, -2.0 / 3.0, 4.0 / 3.0, -2.0 / 3.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, -2.0 / 3.0, -2.0 / 3.0, 4.0 / 3.0))

    val test = bs.penaltyMatrix(1, 16)
    assert(allClose(test, truth, 1e-3))
  }

  test("testDesignMatrix") {
    val U = Vector[Double](0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5)
    val p = 2
    val u = 5.0 / 2.0
    val bs = Bspline((0.0, 5.0), 8, p + 1, U)
    assert(allClose(bs.designMatrix(DenseVector[Double](u), 0),
      DenseVector(0, 0, 0.125, 0.75, 0.125, 0, 0, 0).toDenseMatrix))
    assert(allClose(bs.designMatrix(DenseVector[Double](u), 2),
      DenseVector(0, 0, 1.0, -2.0, 1.0, 0.0, 0.0, 0.0).toDenseMatrix))
  }
}
