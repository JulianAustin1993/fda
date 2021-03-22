package fda
package representation

import basis.Bspline

import breeze.linalg.{DenseMatrix, DenseVector, kron, linspace}
import org.scalatest.FunSuite

class FdTest extends FunSuite {
  val domain: (Double, Double) = (0.0, 10.0)
  val fd: Fd = simulateFd(
    domain = domain,
    meanFunction = x => scala.math.sin(x) + x,
    eigenFunctions = x => DenseVector(scala.math.cos(2.0 * x * scala.math.Pi / 10.0) / 5.0, -1.0 * scala.math.sin(2 * x * scala.math.Pi / 10.0) / 5.0),
    eigenValues = DenseVector(5.0, 2.0),
    sigma2 = 0.1,
    nSubjects = 2000,
    nRegular = 32,
    sparsity = (8, 32))
  val meanBs: Bspline = Bspline(domain, 30, 4)
  val marginalCovBs: Bspline = Bspline(domain, 16, 4)
  val x: DenseVector[Double] = linspace(domain._1, domain._2, 32)
  val Phi: DenseMatrix[Double] = meanBs.designMatrix(x, 0)

  test("testMean") {
    val smoothMean = fd.calculateSmoothMean(meanBs, 2)
    val mu = x.mapValues(t => t + scala.math.sin(t))
    assert(smoothMean.predictMean(Phi).size == mu.size)
  }

  test("testCov") {
    val smoothMean = fd.calculateSmoothMean(meanBs, 2)
    val mu = smoothMean.predictMean(Phi)
    val smoothCov = fd.calculateSmoothCov(marginalCovBs, 2, mu, error = true)
    val X = marginalCovBs.designMatrix(x, 0)
    val XX = kron(X, X)
    assert(smoothCov.predictMean(XX).length == XX.rows)
  }

  test("testSigma2") {
    val smoothMean = fd.calculateSmoothMean(meanBs, 2)
    val mu = smoothMean.predictMean(Phi)
    val smoothCov = fd.calculateSmoothCov(marginalCovBs, 2, mu, error = true)
    val sig2 = fd.calculateSigma2(error = true,
      meanBs,
      marginalCovBs,
      smoothMean,
      2,
      smoothCov)
    assert(scala.math.abs(sig2 - 0.1) < 0.05)
  }

}
