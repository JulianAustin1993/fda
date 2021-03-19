package fda
package kernel

import breeze.linalg.DenseVector
import org.scalatest.FunSuite

class MaternTest extends FunSuite {

  test("testKernel") {
    val (x, y) = (DenseVector(1.0), DenseVector(2.0))
    val kern = Matern(0.5, 1.0, 1.0)
    assert(kern.k(x, y) == scala.math.exp(-1.0))
    assert(kern.k(x, x) == 1.0)
  }
}
