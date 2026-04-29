#include "ExperimentConfig.hh"

#include "G4Exception.hh"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

namespace
{

std::string Trim(const std::string& value)
{
  const auto begin = value.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return "";
  }

  const auto end = value.find_last_not_of(" \t\r\n");
  return value.substr(begin, end - begin + 1);
}

std::string ToLower(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value;
}

std::unordered_map<std::string, std::string> ParseKeyValueFile(
    const std::string& configPath)
{
  std::ifstream input(configPath);
  if (!input) {
    G4Exception("GetExperimentConfig()", "ConfigFileNotFound", FatalException,
                ("Cannot open experiment config: " + configPath).c_str());
  }

  std::unordered_map<std::string, std::string> values;
  std::string line;
  while (std::getline(input, line)) {
    const auto hashPos = line.find('#');
    if (hashPos != std::string::npos) {
      line = line.substr(0, hashPos);
    }

    line = Trim(line);
    if (line.empty()) {
      continue;
    }

    const auto eqPos = line.find('=');
    if (eqPos == std::string::npos) {
      continue;
    }

    auto key = ToLower(Trim(line.substr(0, eqPos)));
    auto value = Trim(line.substr(eqPos + 1));
    values[key] = value;
  }

  return values;
}

std::string GetString(const std::unordered_map<std::string, std::string>& values,
                      const std::string& key,
                      const std::string& fallback)
{
  const auto it = values.find(key);
  return it == values.end() ? fallback : it->second;
}

double GetDouble(const std::unordered_map<std::string, std::string>& values,
                 const std::string& key,
                 double fallback)
{
  const auto it = values.find(key);
  return it == values.end() ? fallback : std::stod(it->second);
}

int GetInt(const std::unordered_map<std::string, std::string>& values,
           const std::string& key,
           int fallback)
{
  const auto it = values.find(key);
  return it == values.end() ? fallback : std::stoi(it->second);
}

long GetLong(const std::unordered_map<std::string, std::string>& values,
             const std::string& key,
             long fallback)
{
  const auto it = values.find(key);
  return it == values.end() ? fallback : std::stol(it->second);
}

std::filesystem::path ResolvePath(const std::filesystem::path& configPath,
                                  const std::string& candidate)
{
  if (candidate.empty()) {
    return {};
  }

  std::filesystem::path path(candidate);
  if (path.is_absolute()) {
    return path.lexically_normal();
  }

  return (configPath.parent_path() / path).lexically_normal();
}

std::filesystem::path ResolveAbsolutePath(const std::string& candidate)
{
  std::filesystem::path path(candidate);
  if (path.empty()) {
    return {};
  }
  if (path.is_absolute()) {
    return path.lexically_normal();
  }
  return std::filesystem::absolute(path).lexically_normal();
}

SourceMode ParseSourceMode(const std::string& rawValue)
{
  const auto value = ToLower(rawValue);
  if (value == "mono" || value == "mono_collimated") {
    return SourceMode::Mono;
  }
  if (value == "phase_space" || value == "phasespace") {
    return SourceMode::PhaseSpace;
  }
  if (value == "spectrum" || value == "spectrum_collimated") {
    return SourceMode::Spectrum;
  }

  G4Exception("GetExperimentConfig()", "InvalidSourceMode", FatalException,
              ("Unknown source_mode: " + rawValue).c_str());
  return SourceMode::Spectrum;
}

OreShape ParseOreShape(const std::string& rawValue)
{
  const auto value = ToLower(rawValue);
  if (value == "slab" || value == "box") {
    return OreShape::Slab;
  }
  if (value == "ellipsoid") {
    return OreShape::Ellipsoid;
  }

  G4Exception("GetExperimentConfig()", "InvalidOreShape", FatalException,
              ("Unknown ore_shape: " + rawValue).c_str());
  return OreShape::Slab;
}

OreMaterialMode ParseOreMaterialMode(const std::string& rawValue)
{
  const auto value = ToLower(rawValue);
  if (value == "single") {
    return OreMaterialMode::Single;
  }
  if (value == "mixture") {
    return OreMaterialMode::Mixture;
  }
  if (value == "air_path" || value == "air" || value == "empty") {
    return OreMaterialMode::AirPath;
  }

  G4Exception("GetExperimentConfig()", "InvalidOreMaterialMode",
              FatalException,
              ("Unknown ore_material_mode: " + rawValue).c_str());
  return OreMaterialMode::Single;
}

HeterogeneityMode ParseHeterogeneityMode(const std::string& rawValue)
{
  const auto value = ToLower(rawValue);
  if (value == "none") {
    return HeterogeneityMode::None;
  }
  if (value == "inclusion") {
    return HeterogeneityMode::Inclusion;
  }

  G4Exception("GetExperimentConfig()", "InvalidHeterogeneityMode",
              FatalException,
              ("Unknown heterogeneity_mode: " + rawValue).c_str());
  return HeterogeneityMode::None;
}

std::string ResolveConfigPath()
{
  const char* envPath = std::getenv("XRT_EXPERIMENT_CONFIG");
  if (envPath != nullptr && envPath[0] != '\0') {
    return envPath;
  }

  return "../source_models/config/experiment_config.txt";
}

ExperimentConfig LoadConfig()
{
  ExperimentConfig config;
  config.configPath = ResolveAbsolutePath(ResolveConfigPath()).string();

  const auto configPath = std::filesystem::path(config.configPath);
  const auto values = ParseKeyValueFile(config.configPath);

  config.runId =
      GetString(values, "run_id", configPath.stem().string());
  config.experimentLabel =
      GetString(values, "experiment_label", "xrt_research_baseline");
  config.outputPrefix =
      GetString(values, "output_prefix", config.experimentLabel);
  config.outputDir = GetString(values, "output_dir", ".");
  config.benchmarkSuite =
      GetString(values, "benchmark_suite", config.benchmarkSuite);
  config.researchRoute =
      GetString(values, "research_route", config.researchRoute);
  config.predictionStage =
      GetString(values, "prediction_stage", config.predictionStage);
  config.runRole =
      GetString(values, "run_role", config.runRole);
  config.sourceVariant =
      GetString(values, "source_variant", config.sourceVariant);
  config.detectorLayout =
      GetString(values, "detector_layout", config.detectorLayout);
  config.prepProfile =
      GetString(values, "prep_profile", config.prepProfile);
  config.feedSizeBand =
      GetString(values, "feed_size_band", config.feedSizeBand);
  config.feedCondition =
      GetString(values, "feed_condition", config.feedCondition);
  config.samplePhotons = GetInt(values, "sample_photons", config.samplePhotons);
  config.randomSeed = GetLong(values, "random_seed", config.randomSeed);

  config.sourceMode =
      ParseSourceMode(GetString(values, "source_mode", "spectrum"));
  config.spectrumFile =
      ResolvePath(configPath, GetString(values, "spectrum_file", "")).string();
  config.phaseSpaceFile =
      ResolvePath(configPath, GetString(values, "phase_space_file", "")).string();
  config.monoEnergyKeV =
      GetDouble(values, "mono_energy_kev", config.monoEnergyKeV);
  config.sourceX_cm = GetDouble(values, "source_x_cm", config.sourceX_cm);
  config.sourceY_mm = GetDouble(values, "source_y_mm", config.sourceY_mm);
  config.sourceZ_mm = GetDouble(values, "source_z_mm", config.sourceZ_mm);
  config.beamHalfY_mm =
      GetDouble(values, "beam_half_y_mm", config.beamHalfY_mm);
  config.beamHalfZ_mm =
      GetDouble(values, "beam_half_z_mm", config.beamHalfZ_mm);
  config.dirX = GetDouble(values, "dir_x", config.dirX);
  config.dirY = GetDouble(values, "dir_y", config.dirY);
  config.dirZ = GetDouble(values, "dir_z", config.dirZ);
  config.incidenceAngleDeg =
      GetDouble(values, "incidence_angle_deg", config.incidenceAngleDeg);

  config.oreMaterialMode = ParseOreMaterialMode(
      GetString(values, "ore_material_mode", "single"));
  config.orePrimaryMaterial = GetString(values, "ore_primary_material",
                                        config.orePrimaryMaterial);
  config.oreSecondaryMaterial = GetString(values, "ore_secondary_material",
                                          config.oreSecondaryMaterial);
  config.oreSecondaryMassFraction = GetDouble(values,
                                              "ore_secondary_mass_fraction",
                                              config.oreSecondaryMassFraction);
  config.oreShape =
      ParseOreShape(GetString(values, "ore_shape", "slab"));
  config.oreThickness_mm =
      GetDouble(values, "ore_thickness_mm", config.oreThickness_mm);
  config.oreHalfY_mm =
      GetDouble(values, "ore_half_y_mm", config.oreHalfY_mm);
  config.oreHalfZ_mm =
      GetDouble(values, "ore_half_z_mm", config.oreHalfZ_mm);

  config.heterogeneityMode = ParseHeterogeneityMode(
      GetString(values, "heterogeneity_mode", "none"));
  config.inclusionMaterial = GetString(values, "inclusion_material",
                                       config.inclusionMaterial);
  config.inclusionShape = ParseOreShape(
      GetString(values, "inclusion_shape", "ellipsoid"));
  config.inclusionThickness_mm =
      GetDouble(values, "inclusion_thickness_mm", config.inclusionThickness_mm);
  config.inclusionRadiusY_mm =
      GetDouble(values, "inclusion_radius_y_mm", config.inclusionRadiusY_mm);
  config.inclusionRadiusZ_mm =
      GetDouble(values, "inclusion_radius_z_mm", config.inclusionRadiusZ_mm);
  config.inclusionOffsetY_mm =
      GetDouble(values, "inclusion_offset_y_mm", config.inclusionOffsetY_mm);
  config.inclusionOffsetZ_mm =
      GetDouble(values, "inclusion_offset_z_mm", config.inclusionOffsetZ_mm);

  config.detectorThickness_mm =
      GetDouble(values, "detector_thickness_mm", config.detectorThickness_mm);
  config.detectorHalfY_mm =
      GetDouble(values, "detector_half_y_mm", config.detectorHalfY_mm);
  config.detectorHalfZ_mm =
      GetDouble(values, "detector_half_z_mm", config.detectorHalfZ_mm);
  config.detectorX_cm =
      GetDouble(values, "detector_x_cm", config.detectorX_cm);
  config.sideDetectorThickness_mm =
      GetDouble(values, "side_detector_thickness_mm",
                config.sideDetectorThickness_mm);
  config.sideDetectorHalfX_mm =
      GetDouble(values, "side_detector_half_x_mm",
                config.sideDetectorHalfX_mm);
  config.sideDetectorHalfZ_mm =
      GetDouble(values, "side_detector_half_z_mm",
                config.sideDetectorHalfZ_mm);
  config.sideDetectorY_cm =
      GetDouble(values, "side_detector_y_cm", config.sideDetectorY_cm);

  config.worldX_cm = GetDouble(values, "world_x_cm", config.worldX_cm);
  config.worldY_cm = GetDouble(values, "world_y_cm", config.worldY_cm);
  config.worldZ_cm = GetDouble(values, "world_z_cm", config.worldZ_cm);
  config.envelopeX_cm =
      GetDouble(values, "envelope_x_cm", config.envelopeX_cm);
  config.envelopeY_cm =
      GetDouble(values, "envelope_y_cm", config.envelopeY_cm);
  config.envelopeZ_cm =
      GetDouble(values, "envelope_z_cm", config.envelopeZ_cm);

  return config;
}

}  // namespace

const ExperimentConfig& GetExperimentConfig()
{
  static const ExperimentConfig config = LoadConfig();
  return config;
}

std::string SourceModeToString(SourceMode mode)
{
  switch (mode) {
    case SourceMode::Mono:
      return "mono";
    case SourceMode::Spectrum:
      return "spectrum";
    case SourceMode::PhaseSpace:
      return "phase_space";
  }

  return "unknown";
}

std::string OreShapeToString(OreShape shape)
{
  switch (shape) {
    case OreShape::Slab:
      return "slab";
    case OreShape::Ellipsoid:
      return "ellipsoid";
  }

  return "unknown";
}

std::string OreMaterialModeToString(OreMaterialMode mode)
{
  switch (mode) {
    case OreMaterialMode::Single:
      return "single";
    case OreMaterialMode::Mixture:
      return "mixture";
    case OreMaterialMode::AirPath:
      return "air_path";
  }

  return "unknown";
}

std::string HeterogeneityModeToString(HeterogeneityMode mode)
{
  switch (mode) {
    case HeterogeneityMode::None:
      return "none";
    case HeterogeneityMode::Inclusion:
      return "inclusion";
  }

  return "unknown";
}

std::string BuildOutputPath(const ExperimentConfig& config,
                            const std::string& suffix)
{
  std::filesystem::path outputDir(config.outputDir);
  if (outputDir.is_relative()) {
    outputDir = std::filesystem::absolute(outputDir);
  }
  outputDir /= config.outputPrefix + suffix;
  return outputDir.lexically_normal().string();
}
