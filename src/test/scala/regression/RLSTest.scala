package fda
package regression

import breeze.linalg.{DenseMatrix, DenseVector, diag, linspace, max, sum}
import breeze.numerics.{abs, sin}
import fda.basis.Bspline
import fda.math.linalg.allClose
import org.scalatest.FunSuite

class RLSTest extends FunSuite {
  val domain: (Double, Double) = (0.0, 10.0)
  val nBasis: Int = 62
  val order: Int = 4
  val x: DenseVector[Double] = linspace(domain._1, domain._2, 64)
  val y: DenseVector[Double] = sin(x*scala.math.Pi * 2.0) + x
  val bs: Bspline = Bspline(domain, nBasis, order)
  val X: DenseMatrix[Double] =bs.designMatrix(x, 0)
  val P: List[DenseMatrix[Double]] = List(bs.penaltyMatrix(2, 16))

  test("testSmooth") {
    val rls = RLS(y, X, P, None, None, None, 1.0, 1000, 1e-8)
    assert(max(abs(rls.residuals))<  1e-3)
  }

  test("testConstraint") {
    val C = DenseMatrix.ones[Double](1, nBasis)
    val rls = RLS(y, X, P, None, None, Option(C), 1.0, 1000, 1e-8)
    assert(scala.math.abs(sum(rls.coefs)) < 1e-8)
  }

  test("testPrediction") {
    val rls = RLS(y, X, P, None, None, None, 1.0, 1000, 1e-8)
    val yhat = rls.predictMean(X)
    assert(allClose(y.toDenseMatrix, yhat.toDenseMatrix, 1e-3))
  }

  test("testPredictionVar") {
    val rls = RLS(y, X, P, None, None, None, 1.0, 1000, 1e-8)
    val v = rls.predictVar(X)
    val vs = diag(rls.predictCov(X))
    assert(allClose(v.toDenseMatrix, vs.toDenseMatrix, 1e-3))
  }

}
