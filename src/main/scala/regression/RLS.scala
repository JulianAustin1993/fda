package fda
package regression

import breeze.linalg.qrp.DenseQRP
import breeze.linalg.{Axis, DenseMatrix, DenseVector, diag, qrp, sum}
import fda.math.linalg.{infNorm, nullSpaceConstraint, oneNorm}


/**
 * Perform regularised least square regression with multiple penalties.
 * @param y response vector
 * @param X Design matrix
 * @param penalties List of penalty matrices
 * @param W Weighting matrix for weigthed least squares.
 * @param H Fixed penalty matrix.
 * @param C Constraint matrix for coefficient constraints.
 * @param gamma Gamma modifier for GCV calculation.
 * @param maxIter Maximum number of iterations for smoothing parameter convergence.
 * @param tol Tolerance for smoothing parameter convergence.
 */
case class RLS(y: DenseVector[Double],
               X: DenseMatrix[Double],
               penalties: List[DenseMatrix[Double]],
               W: Option[DenseMatrix[Double]],
               H: Option[DenseMatrix[Double]],
               C: Option[DenseMatrix[Double]],
               gamma: Double,
               maxIter: Int,
               tol: Double) {
  /*
  Constraint matrix as the null space of C.
   */
  val Z: DenseMatrix[Double] = C match {
    case Some(i) => nullSpaceConstraint(i)
    case None => DenseMatrix.eye[Double](X.cols)
  }

  /*
  Weighting the design matrix
   */
  val WX: DenseMatrix[Double] = W match {
    case Some(i) => i * X
    case None => X
  }

  /*
  Weighting the response vectors.
   */
  val Wy: DenseVector[Double] = W match {
    case Some(i) => i * y
    case None => y
  }

  /*
  Constrained  and weighted design matrix.
   */
  val WXZ: DenseMatrix[Double] = WX * Z

  /*
  Constrained constant penalty matrices.
   */
  val ZtHZ: DenseMatrix[Double] = H match {
    case Some(i) => Z.t * i * Z
    case None => DenseMatrix.zeros[Double](WXZ.cols, WXZ.cols)
  }

  /*
  Pivoted QR decomposition in reduced form of the constrained and weighted design matrix.
   */
  val QRP: DenseQRP = qrp(WXZ)
  val Q = QRP.q(::, 0 until scala.math.min(WXZ.rows, WXZ.cols)).toDenseMatrix
  val R = QRP.r(0 until scala.math.min(WXZ.rows, WXZ.cols), ::).toDenseMatrix
  val P: DenseMatrix[Double] = QRP.pivotMatrix.mapValues(_.toDouble)

  /*
  Normalising and constraining penalty matrices.
   */
  val infnormWXZ: Double = infNorm(WXZ)
  val PtZtPensZP: List[DenseMatrix[Double]] = penalties.map(p => P.t * Z.t * p * Z* P)
  val scaledPtZtPensZP: List[DenseMatrix[Double]] = PtZtPensZP.map(p => {
    p *:* (infnormWXZ / oneNorm(p))
  })

  /*
  Choosing smoothing parameters lambda through minimising the Gcv score.
   */
  private val gcv: Gcv = Gcv(Wy, Q, R, ZtHZ, scaledPtZtPensZP, gamma, maxIter, tol)

  /*
  Minimised smoothing parameter through Gcv.
   */
  val lambda: DenseVector[Double] = gcv.lambda

  /*
   Square root of influence matrix.
   */
  val QU1: DenseMatrix[Double] = Q* gcv.U1

  /*
  influence matrix.
   */
  val influenceMatrix: DenseMatrix[Double] = QU1 * QU1.t

  /*
  Residuals
   */
  val residuals: DenseVector[Double] = Wy - influenceMatrix * Wy

  /*
  Right singular vectors in unconstrained space and changed for pivot.
   */
  val ZPV: DenseMatrix[Double] = Z * P * gcv.Vt.t

  /*
  Estimated coefficients.
   */
  val coefs: DenseVector[Double] = ZPV *gcv.invD * gcv.y1

  /*
  Estimated coefficient variance.
   */
  val coefsCov: DenseMatrix[Double] = (ZPV * gcv.invD * ZPV.t) * gcv.scale

  /*
  Estimated degrees of freedom of smooth.
   */
  val edf: DenseVector[Double] = diag(coefsCov * WX.t * WX) /:/ gcv.scale

  /*
  Estimated noise variance
   */
  val sigma2: Double = gcv.scale

  def predictMean(predictors: DenseMatrix[Double]): DenseVector[Double] = {
    predictors * coefs
  }

  def predictCov(predictors: DenseMatrix[Double]): DenseMatrix[Double] = {
    predictors * coefsCov * predictors.t
  }

  def predictVar(predictors: DenseMatrix[Double]): DenseVector[Double] = {
    val cp = coefsCov * predictors.t
    sum(predictors *:* (coefsCov * predictors.t).t, Axis._1)
  }

}
