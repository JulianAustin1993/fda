package fda
package math.linalg

import breeze.linalg.{DenseMatrix, DenseVector, diag, isClose}
import org.scalatest.FunSuite

class packageTest extends FunSuite {

  test("testIsDiagonal") {
    val d = DenseVector.rand[Double](5)
    assert(isDiagonal(diag(d)))
    val nd = DenseMatrix.rand[Double](5, 5)
    assert(!isDiagonal(nd))

  }

  test("testJitChol") {
    val X = DenseMatrix((1.0, 0.0), (0.0, 0.0))
    val L = jitChol(X, jit_max = 1e-8)
    println(L)
    assert(allClose(L, cholesky(DenseMatrix((1.0, 0.0), (0.0, 1e-18)))))
  }

  test("testNullSpaceConstraint") {
    val C = DenseMatrix.ones[Double](1, 2)
    assert(allClose(nullSpaceConstraint(C),
      DenseMatrix(-0.5 * scala.math.sqrt(2), 0.5 * scala.math.sqrt(2))))
  }

  test("testReducedRankSvd") {
    val D = diag(DenseVector(1.0, 2.0, 3.0, 4.0, 0.0))
    val rrSvD = reducedRankSvd(D)
    assert(rrSvD._2.size == 4)
  }

  test("testCholesky") {
    val sqrtX = DenseMatrix.rand[Double](5, 5)
    val X = sqrtX.t * sqrtX
    val L = cholesky(X)
    assert(allClose(L * L.t, X))
  }

  test("testEigh") {
    val d = DenseVector(1.0, 2.0, 3.0, 4.0)
    val D = diag(d)
    val (eigenValues, eigenVectors) = eigh(D)
    assert(isClose(eigenValues, d))
    assert(allClose(eigenVectors, DenseMatrix.eye[Double](4)))
  }

  test("testChoSolve") {
    val sqrtA = DenseMatrix.rand[Double](5, 5)
    val A = sqrtA.t * sqrtA
    val x = DenseVector.rand[Double](5)
    val b = A * x
    assert(isClose(choSolve(cholesky(A), b.toDenseMatrix.t).toDenseVector, x))
  }

  test("testChoSolve Single RHS") {
    val sqrtA = DenseMatrix.rand[Double](5, 5)
    val A = sqrtA.t * sqrtA
    val x = DenseVector.rand[Double](5)
    val b = A * x
    assert(isClose(choSolve(cholesky(A), b), x))
  }

  test("testColumnKron") {
    val X1 = DenseMatrix((1.0, 2.0), (2.0, 3.0))
    val X2 = DenseMatrix((3.0, 3.0), (4.0, 5.0))
    val X = columnKron(X1, X2)
    val XX = DenseMatrix(
      (1.0 * 3.0, 1.0 * 3.0, 2.0 * 3.0, 2.0 * 3.0),
      (2.0 * 4.0, 2.0 * 5.0, 3.0 * 4.0, 3.0 * 5.0))
    assert(allClose(X, XX))
  }

  test("testBackSolve") {
    val sqrtA = DenseMatrix.rand[Double](5, 5)
    val A = sqrtA.t * sqrtA
    val U = jitChol(A).t
    val x = DenseVector.rand[Double](5)
    val b = U * x
    assert(isClose(backSolve(U, b), x))

  }

  test("testBackSolve multiple RHS") {
    val sqrtA = DenseMatrix.rand[Double](5, 5)
    val A = sqrtA.t * sqrtA
    val U = jitChol(A).t
    val X = DenseMatrix.rand[Double](5, 5)
    val b = U * X
    assert(allClose(backSolve(U, b), X))
  }


}
