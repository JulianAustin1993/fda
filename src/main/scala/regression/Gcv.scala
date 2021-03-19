package fda
package regression


import breeze.linalg.eigSym.DenseEigSym
import breeze.linalg.{DenseMatrix, DenseVector, NotConvergedException, any, diag, eigSym, max, sum, trace}
import breeze.numerics.{abs, exp, pow}
import fda.math.linalg.{jitChol, reducedRankSvd}
import spire.implicits.cfor

import scala.annotation.tailrec

/**
 * Case clas to hold Generalised cross validation minimisation. Immplemnted as in:
 * See. Wood S (2017). Generalized Additive Models: An Introduction with R, 2 edition. Chapman and Hall/CRC.
 * @param y Response vector.
 * @param Q Q matrix of QR decomposition of design matrix.
 * @param R R matrix of QR decomposition of design matrix.
 * @param H Fixed penalty matrix with smoothing parameter 1.0.
 * @param penalties List of penalty matrices each with individual smoothing parameter.
 * @param gamma Gamma value for modifying GCV calculation.
 * @param maxIter Maximum number of iterations to use for convergence.
 * @param tol Tolerance for convergence of GCV.
 */
case class Gcv(y: DenseVector[Double],
               Q: DenseMatrix[Double],
               R: DenseMatrix[Double],
               H: DenseMatrix[Double],
               penalties: List[DenseMatrix[Double]],
               gamma: Double,
               maxIter: Int,
               tol: Double){
  val n: Int = y.size
  private val (nu0, step0, score0) = initialiseGcv()
  private val interimGcvs = minimiseGcv(nu0, step0, score0, 0, maxIter, tol)

  // TODO Check if smoothing parameters are at +-infinity.

  lazy val score: Double = interimGcvs._1
  lazy val alpha: Double = interimGcvs._2
  lazy val delta: Double = interimGcvs._3
  lazy val U1: DenseMatrix[Double] = interimGcvs._4
  lazy val invD: DenseMatrix[Double] = diag(interimGcvs._5.mapValues(x => 1.0/x))
  lazy val Vt: DenseMatrix[Double] = interimGcvs._6
  lazy val y1: DenseVector[Double] = interimGcvs._7
  lazy val scale: Double = interimGcvs._8
  lazy val lambda: DenseVector[Double] = exp(interimGcvs._9)
  lazy val (grad, hess):(DenseVector[Double], DenseMatrix[Double]) = calculateGcvJacAndHess(lambda, alpha, delta, U1, invD, Vt, y1)


  /**
   * Calculate Generalised cross validation score and return interim matrices for use in further functions.
   * @param nu log scale smoothing parameter values.
   * @return Tuple of the:
   *         gcv score,
   *         norm value,
   *         delta value,
   *         U1 matrix,
   *         singular values,
   *         right singular vectors,
   *         y1 vector,
   *         scale or estimated error variance,
   *         log smoothing parameters.
   */
  private def calculateGcv(nu: DenseVector[Double]): (Double, Double, Double, DenseMatrix[Double], DenseVector[Double], DenseMatrix[Double], DenseVector[Double], Double, DenseVector[Double]) = {
    val thetas: List[Double] = exp(nu).toScalaVector().toList
    val P: DenseMatrix[Double] = penalties.zip(thetas).map(a => a._1 *:* a._2).fold(H)((a, b) => a + b)
    val B: DenseMatrix[Double] = jitChol(P, 1e-16, 1e-5).t
    val rRSvd: (DenseMatrix[Double], DenseVector[Double], DenseMatrix[Double]) = reducedRankSvd(DenseMatrix.vertcat(R, B))
    val U1: DenseMatrix[Double] = rRSvd._1(0 until R.rows, ::).toDenseMatrix
    val U1U1t: DenseMatrix[Double] = U1 * U1.t
    val A: DenseMatrix[Double] = Q * U1U1t * Q.t
    val trA: Double = trace(U1U1t)
    val alpha: Double = sum(pow(y - (A * y), 2))
    val y1: DenseVector[Double] = U1.t * Q.t * y
    val delta: Double = n - gamma * trA
    val scale: Double = alpha / (n - trA)
    val d: DenseVector[Double] = rRSvd._2
    val Vt: DenseMatrix[Double] = rRSvd._3
    val score: Double = n * alpha / scala.math.pow(delta , 2)
    (score, alpha, delta, U1, d, Vt, y1, scale, nu)
  }

  /**
   * Calculate the Jacobian and the hessian of the generalised cross validation wrt to log scale parameters.
   * @param lambdas smoothing parameters.
   * @param alpha norm value from Gcv calculation.
   * @param delta delta value from Gcv calculation.
   * @param U1 U1 matrix from Gcv calculation
   * @param invD Inverse of singular values matrix.
   * @param Vt Right singular vector from Gcv calculation.
   * @param y1 y1 matrix from Gcv calculation
   * @return The jacobian and hessian of the Gcv function at theta
   */
  private def calculateGcvJacAndHess(lambdas: DenseVector[Double],
                                     alpha: Double,
                                     delta: Double,
                                     U1: DenseMatrix[Double],
                                     invD: DenseMatrix[Double],
                                     Vt: DenseMatrix[Double],
                                     y1: DenseVector[Double]): (DenseVector[Double], DenseMatrix[Double]) ={
    val xx = n.toDouble / scala.math.pow(delta,2)
    val xx1 = xx * 2 * alpha / delta
    val x1 = -2.0 * xx / delta
    val x2 = 3.0 * xx1 / delta
    val U1tU1 = U1.t * U1
    val Ms = penalties.map(p => invD * Vt * p * Vt.t * invD)
    val Ks = Ms.map(m => m * U1tU1)
    val thetasList = lambdas.toArray.toList
    val gradAlpha = thetasList.zip(Ms.zip(Ks)).map(a => 2.0 * a._1 * (y1.t * (a._2._1 - a._2._2)* y1))
    val gradDelta = thetasList.zip(Ks).map(a => gamma * a._1 * trace(a._2))
    val gradGcv= xx * DenseVector(gradAlpha:_*) - xx1 * DenseVector(gradDelta:_*)
    val p = gradGcv.size
    val hessGcv = DenseMatrix.zeros[Double](p, p)
    cfor(0)(_ < p, _ + 1){
      i => {
        cfor(0)(_ < i+1, _ + 1){
          j => {
            val mult = 2.0 * lambdas(i) * lambdas(j)
            val MjKi = Ms(j) * Ks(i)
            val tmp = (Ms(i) * Ks(j)) + MjKi - (Ms(i) * Ms(j)) - (Ms(j) * Ms(i)) + (Ks(i)*Ms(j))
            val hA =  mult * y1.t * tmp * y1
            val hD = -gamma * mult * trace(MjKi)
            hessGcv(i, j) = x1 * (gradDelta(j)*gradAlpha(i) + gradDelta(i)*gradAlpha(i)) + (xx * hA(0)) + (x2 * gradDelta(i) * gradDelta(j)) - (xx1 * hD)
            hessGcv(j, i) = hessGcv(i,j)
          }
        }
      }
    }
    diag(hessGcv) := diag(hessGcv) + gradGcv
    (gradGcv, hessGcv)
  }

  /**
   * Initialise the first smoothing parameter value on log scale with initial Gcv score.
   * @return log scale smoothing parameter, first step direction and score.
   */
  private def initialiseGcv(): (DenseVector[Double], DenseVector[Double], Double) ={
    val nu: DenseVector[Double] = DenseVector(penalties.map(p => {
      val rs = R * p * R.t
      scala.math.pow(trace(rs), -1) / n
    }):_*)
    val interimGcvs = calculateGcv(nu)
    val Vt = interimGcvs._6
    val invD = diag(interimGcvs._5.mapValues(x => 1.0/x))
    val y1 = interimGcvs._7
    val scale = interimGcvs._8
    val b: DenseVector[Double] = Vt.t * invD * y1
    val step: DenseVector[Double] = DenseVector(penalties.map(p => scala.math.log(scale * p.rows/(b.t * p * b))):_*) - nu
    (nu, step, interimGcvs._1)
  }

  /**
   * Minimise the Gcv score by adjusting smoothing parameters in log scale.
   * @param nu Current Log scale smoothing parameters.
   * @param step Step direction to take for new smoothing parameter.
   * @param minScore Current minimum Gcv score.
   * @param iteration Current iteration.
   * @param maxIts Maximum iterations to allow before not converged exception.
   * @param tol Tolerance to use for convergence check.
   * @return Tuple of the:
   *         gcv score,
   *         norm value,
   *         delta value,
   *         U1 matrix,
   *         singular values,
   *         right singular vectors,
   *         y1 vector,
   *         scale or estimated error variance,
   *         log smoothing parameters.
   */
  @tailrec
  private final def minimiseGcv(nu: DenseVector[Double],
                                step: DenseVector[Double],
                                minScore: Double,
                                iteration: Int,
                                maxIts: Int,
                                tol: Double): (Double, Double, Double, DenseMatrix[Double], DenseVector[Double], DenseMatrix[Double], DenseVector[Double], Double, DenseVector[Double]) ={
    if (iteration > maxIts)
      throw new NotConvergedException(NotConvergedException.Iterations)
    else {
      var tries = 0
      var s = step
      var interimGcvs = calculateGcv(nu + s)
      var newScore = interimGcvs._1

      while(newScore > minScore & tries < 4){
        s = s /:/ 2.0
        interimGcvs = calculateGcv(nu + s)
        newScore = interimGcvs._1
        tries += 1
      }
      if (tries == 4 & iteration > 3){
        return interimGcvs
      }
      val newNu = nu + s
      val invD = diag(interimGcvs._5.mapValues(x => 1.0/x))
      val (gradGcv, hessGcv) = calculateGcvJacAndHess(exp(newNu),
        interimGcvs._2,
        interimGcvs._3,
        interimGcvs._4,
        invD,
        interimGcvs._6,
        interimGcvs._7)
      if (minScore - newScore < tol * (1 + minScore) & iteration > 3){
        return interimGcvs
      }
      if (scala.math.sqrt(gradGcv.t * gradGcv) <= scala.math.pow(tol ,1.0 / 3.0) * (1 + scala.math.abs(minScore)) & iteration > 3) {
        interimGcvs
      } else {
        val newStep = {
          val eigen: DenseEigSym = eigSym(hessGcv)
          if (any(eigen.eigenvalues <:< 0.0)) {
            - gradGcv / max(abs(gradGcv))
          } else {
            val possStep = - eigen.eigenvectors * diag(eigen.eigenvalues.mapValues(w => 1.0 / w)) * eigen.eigenvectors.t * gradGcv
            val maxStepDir = max(abs(possStep))
            if (maxStepDir > 5.0) possStep * 5.0 / maxStepDir else possStep
          }
        }
        minimiseGcv(newNu, newStep, scala.math.min(newScore, minScore), iteration + 1, maxIts, tol)
      }
    }
  }


}
