package fda
package math.mcmc

import breeze.linalg.DenseVector
import breeze.stats.distributions.{Rand, RandBasis}
import spire.implicits.cfor

trait MH {
  def logLikelihood(x: DenseVector[Double]): Double

  def logTransitionProbability(start: DenseVector[Double], end: DenseVector[Double]): Double

  def proposalDraw(x: DenseVector[Double]): DenseVector[Double]

  def likelihood(x: DenseVector[Double]): Double = scala.math.exp(logLikelihood(x))

  def likelihoodRatio(start: DenseVector[Double], end: DenseVector[Double]): Double = scala.math.exp(logLikelihoodRatio(start, end))

  def logLikelihoodRatio(start: DenseVector[Double], end: DenseVector[Double]): Double =
    logLikelihood(end) - logLikelihood(start) - logTransitionProbability(start, end) + logTransitionProbability(
      end,
      start)

  def rand: RandBasis

  protected def nextDouble: Double = this.rand.generator.nextDouble
}

abstract class BaseMH(
                       logLikelihoodFunc: DenseVector[Double] => Double,
                       init: DenseVector[Double],
                       burnIn: Long = 0,
                       dropCount: Int = 0
                     )(implicit val rand: RandBasis = Rand) extends MH {
  private var last: DenseVector[Double] = init

  def logLikelihood(x: DenseVector[Double]): Double = logLikelihoodFunc(x)

  def draw: DenseVector[Double] = {
    if (dropCount == 0) {
      getNext
    } else {
      cfor(0)(i => i < dropCount, i => i + 1)(i => getNext)
      getNext
    }
  }

  cfor(0)(i => i < burnIn, i => i + 1)(i => getNext)

  private def getNext: DenseVector[Double] = {
    val maybeNext = proposalDraw(last)
    val logAcceptanceRatio = logLikelihoodRatio(last, maybeNext)
    if (logAcceptanceRatio > 0.0) { //This is logically unnecessary, but allows us to skip a call to nextDouble
      last = maybeNext
      maybeNext
    } else {
      if (scala.math.log(nextDouble) < logAcceptanceRatio) {
        last = maybeNext
        maybeNext
      } else {
        last
      }
    }
  }
}

case class MetropolisHastings(
                               logLikelihood: DenseVector[Double] => Double,
                               proposal: DenseVector[Double] => Rand[DenseVector[Double]],
                               logProposalDensity: (DenseVector[Double], DenseVector[Double]) => Double,
                               init: DenseVector[Double],
                               burnIn: Long = 0,
                               dropCount: Int = 0)(implicit rand: RandBasis = Rand)
  extends BaseMH(logLikelihood, init, burnIn, dropCount)(rand) {
  override def proposalDraw(x: DenseVector[Double]): DenseVector[Double] = proposal(x).draw()

  override def logTransitionProbability(start: DenseVector[Double], end: DenseVector[Double]): Double = logProposalDensity(start, end)
}

case class AffineStepMetropolisHastings(
                                         logLikelihood: DenseVector[Double] => Double,
                                         proposalStep: Rand[DenseVector[Double]],
                                         init: DenseVector[Double],
                                         burnIn: Long = 0,
                                         dropCount: Int = 0)(implicit rand: RandBasis = Rand)
  extends BaseMH(logLikelihood, init, burnIn, dropCount)(rand) {
  override def logTransitionProbability(start: DenseVector[Double], end: DenseVector[Double]): Double = 0.0

  override def proposalDraw(x: DenseVector[Double]): DenseVector[Double] = x + proposalStep.draw()
}