package fda
package representation

import breeze.linalg.{*, Axis, DenseMatrix, DenseVector, diag, kron, linspace, sum}
import fda.basis.Basis
import fda.math.integration.rombergIntegrator
import fda.math.linalg.columnKron
import fda.regression.RLS
import spire.implicits.cfor

import scala.Double.NaN

/**
 * Class to hold functional observations and provide acces to rough mean and covariance estimates.
 * @param lY: List of observed responses. Each element is a separate functional observation.
 * @param lX: List of observation points. Each element is the points of observation for separate functional observation.
 * @param regularX: DenseVector of common regular observation points.
 */
case class Fd(lY: List[DenseVector[Double]],
              lX: List[DenseVector[Double]],
              regularX: DenseVector[Double]){

  /*
  Number of subjects.
   */
  val nSubjects = lY.size

  /*
  Number of common observations.
   */
  val nRegular = regularX.size

  /*
  Interim values to get sum and counts.
   */
  private val listRegularX = regularX.toArray.toList
  private val (listRegularYSum, listRegularYCount)= lY.zip(lX).map(a => {
    val r = DenseVector.zeros[Double](nRegular)
    val counts = DenseVector.zeros[Double](nRegular)
    cfor(0)(_ < a._2.size, _ + 1) {
      i => {
        val xi: Double = a._2(i)
        val j: Int = listRegularX.map(t => scala.math.pow(t-xi,2)).zipWithIndex.min._2
        r(j) += a._1(i)
        counts(j) += 1.0
      }
    }
    (r.toArray, counts.toArray)
  }).unzip

  /*
  Interim matrices of sum and counts on regular grid.
   */
  private val regularYSum = DenseMatrix(listRegularYSum:_*)
  private val regularYCount = DenseMatrix(listRegularYCount:_*)

  /*
  All functional observations on common grid, binned to closes point in regularX.
  NaN represents no observations for that grid point.
   */
  val regularY: DenseMatrix[Double] = regularYSum /:/ regularYCount

  /*
  Raw mean vector of observed functional variables.
   */
  lazy val rawMean: DenseVector[Double] = (sum(regularYSum, Axis._0)/:/sum(regularYCount, Axis._0)).t

  /**
   * Calculate the raw covariance matrix with estimated mean mu.
   * @param mu: estimated mean vector on regularX.
   * @return Estimated raw covariance values on grid regularX times regularX.
   */
  def rawCov(mu: DenseVector[Double]): DenseMatrix[Double] = {
    val centredRegularY = (regularY(*, ::) - mu).mapValues {
      case x if x.isNaN => 0.0
      case x => x
    }
    val covSum = centredRegularY.t * centredRegularY
    val covCount = regularYCount.t * regularYCount
    covSum /:/ covCount
  }

  /**
   * Calculate the raw variance matrix with estimated mean mu.
   * @param mu: estiimated mean vector on regularX.
   * @return Estimated raw variance values on grid regularX times regularX.
   */
  def rawVar(mu: DenseVector[Double]): DenseVector[Double] = diag(rawCov(mu))

  /**
   * Obtain a smooth of the mean function from functional data.
   * @param basis Basis system to use for regularised smooth.
   * @param penaltyOrder Penalty order to use for the smooth.
   * @return RLS
   */
  def calculateSmoothMean(basis: Basis,
                 penaltyOrder: Int): RLS = {
    val gamma = if (nRegular < 1000) 1.3 else 2.0
    val Phi = basis.designMatrix(regularX, 0)
    val (listY, listInds) = rawMean.toArray.zipWithIndex.filter(x => !x._1.isNaN).unzip
    val y = DenseVector(listY:_*)
    val X = Phi(listInds.toIndexedSeq, ::).toDenseMatrix
    RLS(y, X, List(basis.penaltyMatrix(penaltyOrder, 16)), None, None, None, gamma, 1000, 1e-6)
  }

  /**
   * Obtain a smooth of the covariance surface from functional data with low rank scale invariant tensor product smooth.
   * @param marginalBasis marginal basis system over single axis of the covariance surface.
   * @param penaltyOrder The order of penalty to use in smooth.
   * @param smoothMean: Vector of smoothed mean on regular grid.
   * @param error Whether to assume data has observation error.
   * @return RLS
   */
  def calculateSmoothCov(marginalBasis: Basis,
                   penaltyOrder: Int,
                   smoothMean: DenseVector[Double],
                   error: Boolean): RLS = {
    val I = DenseMatrix.eye[Double](marginalBasis.nComponents)
    val penalty = marginalBasis.penaltyMatrix(penaltyOrder, 16)
    val P = List(kron(penalty, I), kron(I, penalty))
    val rC: DenseVector[Double] = if (error) {
      val rCFull = rawCov(smoothMean)
      diag(rCFull) := NaN
      rCFull.toDenseVector
    } else {
      rawCov(smoothMean).toDenseVector
    }
    val (listY, listInds) = rC.toArray.zipWithIndex.filter(x => !x._1.isNaN).unzip
    val y: DenseVector[Double] = DenseVector(listY:_*)
    val Phi = marginalBasis.designMatrix(regularX,0)
    val X = kron(Phi, Phi)
    val gamma = if (y.size < 1000) 1.3 else 2.0
    RLS(y, X(listInds.toIndexedSeq, ::).toDenseMatrix, P, None, None, None, gamma, 1000, 1e-6)
  }


  /**
   * Obtain noise estimate from functional data.
   *
   * @param error Whether to assume observation error is in model.
   * @param varBasis Basis system for smooth of variance.
   * @param marginalCovBasis Basis system used for covariance smooth.
   * @param mu: RLS for mean function.
   * @param penaltyOrder: Order to form the penalty matrix of variance function.
   * @param cov RLS for covariance surface.
   * @return Estimated observation error variance.
   */
  def calculateSigma2(error:Boolean,
                      varBasis: Basis,
                      marginalCovBasis: Basis,
                      mu: RLS,
                      penaltyOrder: Int,
                      cov: RLS): Double ={
    if (error) {
      val V = {
        val gamma = if (nRegular < 1000) 1.3 else 2.0
        val Phi = varBasis.designMatrix(regularX, 0)
        val (listY, listInds) = rawVar(mu.predictMean(Phi)).toArray.zipWithIndex.filter(x => !x._1.isNaN).unzip
        val y = DenseVector(listY:_*)
        val X = Phi(listInds.toIndexedSeq, ::).toDenseMatrix
        val P = varBasis.penaltyMatrix(penaltyOrder, 16)
        RLS(y, X, List(P), None, None, None, gamma, 1000, 1e-8)
      }
      val domainRange = varBasis.domain._2 - varBasis.domain._1
      val k = 9
      val J = scala.math.pow(2, k).toInt
      val varX = linspace(varBasis.domain._1 + 0.25 * domainRange, varBasis.domain._1 + 0.75 * domainRange, J+1)
      val X = varBasis.designMatrix(varX,0)
      val covX = marginalCovBasis.designMatrix(varX, 0)
      val XX = columnKron(covX, covX)
      val integrand = V.predictMean(X) - cov.predictMean(XX)
      rombergIntegrator(integrand, k, domainRange/J) / (0.5 * domainRange)
    } else {
      1e-6
    }
  }
}
