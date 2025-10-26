// Further Enhanced example usage of the body_analyzer package.
// Building on previous version: Added CLI support with arg parsing (using built-in args),
// unit test-like assertions, CSV export, performance timing, logging with simple logger,
// and async simulation for API calls. Assumes no additional deps beyond dart:core.
// For full CLI: dart run example_usage.dart --mode=full --output=csv
// Run with: dart run example_usage.dart

import 'dart:convert';
import 'dart:io';

import 'package:body_analyzer/body_analyzer.dart'; // Import the package

// ANSI color codes for enhanced visual appeal (terminal support assumed).
const String _reset = '\u001B[0m';
const String _bold = '\u001B[1m';
const String _green = '\u001B[32m';
const String _yellow = '\u001B[33m';
const String _red = '\u001B[31m';
const String _blue = '\u001B[34m';
const String _cyan = '\u001B[36m';

// Simple logger class for enhanced debugging with colors.
class SimpleLogger {
  static void info(String message) => print('${_cyan}ℹ️  INFO:$_reset $message');
  static void warn(String message) => print('${_yellow}⚠️  WARN:$_reset $message');
  static void error(String message) => print('${_red}❌ ERROR:$_reset $message');
  static void debug(String message) => print('${_blue}🔍 DEBUG:$_reset $message');
}

// CLI argument parser (simplified using built-in parsing).
class CliArgs {
  final String mode;
  final String output;
  final bool verbose;

  CliArgs(List<String> args)
      : mode = _parseArg(args, '--mode', 'full'),
        output = _parseArg(args, '--output', 'json'),
        verbose = args.contains('--verbose');

  static String _parseArg(List<String> args, String flag, String defaultValue) {
    final index = args.indexWhere((a) => a == flag);
    return index != -1 && index + 1 < args.length ? args[index + 1] : defaultValue;
  }
}

// Enhanced helper for safe parsing with logging.
double? safeParseDouble(String? input, {required double minValue, String field = ''}) {
  if (input == null || input.isEmpty) {
    SimpleLogger.warn('Empty input for $field');
    return null;
  }
  final value = double.tryParse(input.trim());
  if (value == null || value < minValue) {
    SimpleLogger.warn('Invalid value for $field: $input (must be >= $minValue)');
    return null;
  }
  return value;
}

// Modular function for basic calculations with timing.
Future<void> demonstrateBasicCalculations({bool verbose = false}) async {
  final stopwatch = Stopwatch()..start();
  SimpleLogger.info('Starting basic calculations...');
  final breastCalc = BreastSizeCalculator();
  try {
    final result1 = breastCalc.calculateSize(36.0, 32.0);
    print('${_green}👚 $result1$_reset'); // Output: Breast Size: 32B
    if (verbose) SimpleLogger.debug('Breast calc successful');
  } catch (e) {
    SimpleLogger.error('Error in breast calculation: $e');
  }

  final buttCalc = ButtSizeCalculator();
  final result2 = buttCalc.calculateSize(38.5, 28.0);
  print('${_green}🍑 $result2$_reset'); // Output: Butt Size Category: Medium (Hip: 38.5")

  final waistCalc = WaistSizeCalculator();
  final result3 = waistCalc.calculateSize(30.0, 65.0);
  print('${_green}📏 $result3$_reset'); // Output: Waist Size Category: Healthy (Small) (Waist: 30.0")
  stopwatch.stop();
  SimpleLogger.info('Basic calculations completed in ${stopwatch.elapsedMilliseconds}ms');
}

// Modular function for advanced calculations with assertions.
void demonstrateAdvancedCalculations({bool verbose = false}) {
  print('\n${_bold}${_cyan}${'═' * 60}$_reset');
  print('${_bold}${_yellow}🌟 ADVANCED CALCULATIONS (Ratios & Limbs) 🌟$_reset');
  print('${_bold}${_cyan}${'═' * 60}$_reset');
  final buttCalc = ButtSizeCalculator(); // Reused instance
  final whr = buttCalc.calculateWHR(28.0, 38.5);
  print('${_green}💚 $whr$_reset'); // Output: Waist-to-Hip Ratio: 0.73 (Low risk (Healthy))

  // Assertion for validation
  assert(whr.contains('0.73'), 'WHR calculation mismatch');
  if (verbose) SimpleLogger.debug('WHR assertion passed');

  // Example with larger measurements for diversity
  final whrLarge = buttCalc.calculateWHR(35.0, 42.0);
  print('${_yellow}🟡 $whrLarge$_reset'); // Output: Waist-to-Hip Ratio: 0.83 (Low risk (Healthy))

  final armCalc = ArmSizeCalculator();
  final armResult = armCalc.calculateSize(13.0, 12.5);
  print('${_green}💪 $armResult$_reset'); // Output: Arm Size Category: Medium (Avg: 12.8")

  final thighCalc = ThighSizeCalculator();
  final thighResult = thighCalc.calculateSize(22.0, 14.0);
  print('${_green}🦵 $thighResult$_reset'); // Output: Thigh Size Category: Medium (Thigh: 22.0")
  print('${_bold}${_cyan}${'═' * 60}$_reset\n');
}

// Modular function for full profile with CSV export option.
Future<void> demonstrateFullProfile({bool verbose = false, String output = 'json'}) async {
  print('\n${_bold}${_cyan}${'═' * 70}$_reset');
  print('${_bold}${_blue}📊 FULL BODY PROFILE ANALYSIS 📊$_reset');
  print('${_bold}${_cyan}${'═' * 70}$_reset');
  final profile = BodyProfile(
    bust: 36.0,
    underbust: 32.0,
    hip: 38.5,
    waist: 28.0,
    height: 65.0,
  );

  // Get comprehensive profile
  final fullProfile = profile.getFullProfile();
  print('${_yellow}🔹 Standard Profile:$_reset');
  fullProfile.forEach((key, value) => print('   ${_green}$key: $value$_reset'));

  // Diverse example: Larger build profile
  final largeProfile = BodyProfile(
    bust: 42.0,
    underbust: 38.0,
    hip: 44.0,
    waist: 34.0,
    height: 68.0,
  );
  final largeFullProfile = largeProfile.getFullProfile();
  print('\n${_yellow}🔹 Diverse Example (Larger Build):$_reset');
  largeFullProfile.forEach((key, value) => print('   ${_green}$key: $value$_reset'));

  // Estimate BMI with error handling
  try {
    final bmi = profile.estimateBMI(60.0);
    print('\n${_bold}${_yellow}⚖️  $bmi$_reset'); // Output: Estimated BMI: 21.7 (Normal)
    assert(bmi.contains('21.7'), 'BMI calculation mismatch');
  } catch (e) {
    SimpleLogger.error('Error in BMI estimation: $e');
  }

  // Calculate health score
  final healthScore = HealthScoreAggregator.calculateHealthScore(profile, 60.0);
  print('${_bold}${_green}❤️ Health Score: ${healthScore.toStringAsFixed(1)}/100$_reset'); // Output: Health Score: 100.0/100

  // Get clothing recommendations
  final clothingRec = getClothingSizeRecommendation(profile);
  print('${_bold}${_yellow}👗 $clothingRec$_reset'); // Output: Recommended Sizes - Top: S, Bottom: 4-6 (S/M), Dress: S

  // Unit conversion with full output
  final cmMeasurements = profile.convertToCm();
  final formattedCm = <String, String>{for (var e in cmMeasurements.entries) e.key: e.value.toStringAsFixed(1)};
  print('${_bold}${_cyan}📏 Full measurements in cm: ${jsonEncode(formattedCm)}$_reset');

  // Reverse conversion example
  final reversedInches = profile.convertToInches(cmMeasurements);
  print('${_bold}${_green}🔄 Reversed to inches (bust): ${reversedInches['bust']!.toStringAsFixed(1)}$_reset'); // Matches original

  // Export to CSV if requested
  if (output == 'csv') {
    await _exportToCsv(profile, fullProfile, healthScore, clothingRec);
  }
  print('${_bold}${_cyan}${'═' * 70}$_reset\n');
}

// Helper for CSV export.
Future<void> _exportToCsv(BodyProfile profile, Map<String, String> fullProfile, double healthScore, String clothingRec) async {
  final csvHeader = 'Metric,Value\n';
  final csvRows = <String>[];
  fullProfile.forEach((key, value) => csvRows.add('$key,$value'));
  csvRows.add('health_score,${healthScore.toStringAsFixed(1)}');
  csvRows.add('clothing,$clothingRec');
  final csvContent = csvHeader + csvRows.join('\n');
  
  final file = File('profile_analysis.csv');
  await file.writeAsString(csvContent);
  SimpleLogger.info('Profile exported to CSV: ${file.path}');
}

// Enhanced serialization with versioning and timestamps.
Map<String, dynamic> serializeProfileToMap(BodyProfile profile, double weightKg, Map<String, String> fullProfile, double healthScore) {
  return {
    'version': '2.0',
    'timestamp': DateTime.now().toIso8601String(),
    'profile': {
      'bust': profile.bust,
      'underbust': profile.underbust,
      'hip': profile.hip,
      'waist': profile.waist,
      'height': profile.height,
      'weight_kg': weightKg,
    },
    'analysis': fullProfile,
    'bmi': profile.estimateBMI(weightKg),
    'health_score': healthScore,
    'clothing_recommendations': getClothingSizeRecommendation(profile),
    'cm_conversion': profile.convertToCm(),
  };
}

// Simulate async API call with JSON.
Future<String> simulateApiCall(Map<String, dynamic> data) async {
  // Simulate network delay
  await Future.delayed(const Duration(milliseconds: 500));
  return jsonEncode(data);
}

// Demonstrate integration with async simulation.
Future<void> demonstrateIntegration({bool verbose = false}) async {
  print('\n${_bold}${_cyan}${'═' * 60}$_reset');
  print('${_bold}${_blue}🔌 ENHANCED API/DATABASE INTEGRATION (Async JSON + Export) 🔌$_reset');
  print('${_bold}${_cyan}${'═' * 60}$_reset');
  final profile = BodyProfile(
    bust: 36.0,
    underbust: 32.0,
    hip: 38.5,
    waist: 28.0,
    height: 65.0,
  );
  final fullProfile = profile.getFullProfile();
  final healthScore = HealthScoreAggregator.calculateHealthScore(profile, 60.0);
  
  final dataMap = serializeProfileToMap(profile, 60.0, fullProfile, healthScore);
  print('${_yellow}⏳ Simulating API call...$_reset');
  final jsonOutput = await simulateApiCall(dataMap);
  print('${_bold}${_green}📡 Simulated API Response (snippet): ${jsonOutput.substring(0, 200)}...$_reset');

  // Save to file
  final file = File('profile_analysis.json');
  await file.writeAsString(jsonOutput);
  SimpleLogger.info('Async profile saved to: ${file.path}');

  if (verbose) {
    SimpleLogger.debug('Full data map keys: ${dataMap.keys.toList()}');
  }
  print('${_bold}${_cyan}${'═' * 60}$_reset\n');
}

// Async main for better structure.
Future<void> main(List<String> args) async {
  final cliArgs = CliArgs(args);
  print('${_bold}${_yellow}🎉 ${'═' * 50} 🎉$_reset');
  SimpleLogger.info('🚀 Starting Further Enhanced body_analyzer Package Demo');
  SimpleLogger.info('Mode: ${cliArgs.mode}, Output: ${cliArgs.output}, Verbose: ${cliArgs.verbose}');
  print('${_bold}${_green}📅 Current Date: ${DateTime.now().toString().split(' ')[0]} - Version: 2.1 Further Enhanced$_reset');
  print('${_bold}${_yellow}🎉 ${'═' * 50} 🎉$_reset');

  await demonstrateBasicCalculations(verbose: cliArgs.verbose);
  demonstrateAdvancedCalculations(verbose: cliArgs.verbose);
  await demonstrateFullProfile(verbose: cliArgs.verbose, output: cliArgs.output);
  await demonstrateIntegration(verbose: cliArgs.verbose);

  print('\n${_bold}${_red}🏁 ${'═' * 50} 🏁$_reset');
  print('${_bold}${_cyan}🌟 Demo Complete! 🌟$_reset');
  print('This enhanced example includes CLI args, async ops, CSV export, and assertions for robust production use.');
  print('${_yellow}💡 CLI Usage: dart run example_usage.dart --mode=full --output=csv --verbose$_reset');
  print('${_bold}${_red}🏁 ${'═' * 50} 🏁$_reset');
}
