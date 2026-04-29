#ifndef XRTExperimentConfig_h
#define XRTExperimentConfig_h 1

#include <string>

enum class SourceMode
{
  Mono,
  Spectrum,
  PhaseSpace
};

enum class OreShape
{
  Slab,
  Ellipsoid
};

enum class OreMaterialMode
{
  Single,
  Mixture,
  AirPath
};

enum class HeterogeneityMode
{
  None,
  Inclusion
};

struct ExperimentConfig
{
  std::string configPath;
  std::string runId;
  std::string experimentLabel;
  std::string outputPrefix;
  std::string outputDir;
  std::string benchmarkSuite = "sim_only_v1";
  std::string researchRoute = "lab";
  std::string predictionStage = "fine";
  std::string runRole = "material";
  std::string sourceVariant = "normal_narrow";
  std::string detectorLayout = "transmission";
  std::string prepProfile = "medium_prep";
  std::string feedSizeBand = "standard_block";
  std::string feedCondition = "clean_dry_single_piece";
  int samplePhotons = 100;
  long randomSeed = -1;

  SourceMode sourceMode = SourceMode::Spectrum;
  std::string spectrumFile;
  std::string phaseSpaceFile;
  double monoEnergyKeV = 80.0;
  double sourceX_cm = -30.0;
  double sourceY_mm = 0.0;
  double sourceZ_mm = 0.0;
  double beamHalfY_mm = 5.0;
  double beamHalfZ_mm = 5.0;
  double dirX = 1.0;
  double dirY = 0.0;
  double dirZ = 0.0;
  double incidenceAngleDeg = 0.0;

  OreMaterialMode oreMaterialMode = OreMaterialMode::Single;
  std::string orePrimaryMaterial = "Quartz";
  std::string oreSecondaryMaterial = "Magnetite";
  double oreSecondaryMassFraction = 0.0;
  OreShape oreShape = OreShape::Slab;
  double oreThickness_mm = 10.0;
  double oreHalfY_mm = 100.0;
  double oreHalfZ_mm = 100.0;

  HeterogeneityMode heterogeneityMode = HeterogeneityMode::None;
  std::string inclusionMaterial = "Magnetite";
  OreShape inclusionShape = OreShape::Ellipsoid;
  double inclusionThickness_mm = 4.0;
  double inclusionRadiusY_mm = 20.0;
  double inclusionRadiusZ_mm = 20.0;
  double inclusionOffsetY_mm = 0.0;
  double inclusionOffsetZ_mm = 0.0;

  double detectorThickness_mm = 5.0;
  double detectorHalfY_mm = 100.0;
  double detectorHalfZ_mm = 100.0;
  double detectorX_cm = 25.0;
  double sideDetectorThickness_mm = 5.0;
  double sideDetectorHalfX_mm = 120.0;
  double sideDetectorHalfZ_mm = 100.0;
  double sideDetectorY_cm = 12.0;

  double worldX_cm = 100.0;
  double worldY_cm = 50.0;
  double worldZ_cm = 50.0;
  double envelopeX_cm = 80.0;
  double envelopeY_cm = 30.0;
  double envelopeZ_cm = 30.0;
};

const ExperimentConfig& GetExperimentConfig();

std::string SourceModeToString(SourceMode mode);
std::string OreShapeToString(OreShape shape);
std::string OreMaterialModeToString(OreMaterialMode mode);
std::string HeterogeneityModeToString(HeterogeneityMode mode);

std::string BuildOutputPath(const ExperimentConfig& config,
                            const std::string& suffix);

#endif
