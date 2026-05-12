/**
 * ============================================================================
 *  RECOMMENDATION UI CONTROLLER
 *  Handles all DOM manipulation and user interaction for the ML feature
 * ============================================================================
 *  This file is separate from recommendation.js to follow good coding practices:
 *    - recommendation.js  = pure ML logic (no DOM dependencies)
 *    - recommendation-ui.js = UI rendering and event handling
 * ============================================================================
 */

// Wait for the page to fully load before initializing
document.addEventListener("DOMContentLoaded", function () {

  // ========================================================================
  // POPULATE THE HOUSEHOLD SELECTOR DROPDOWN
  // ========================================================================
  const householdSelect = document.getElementById("household-select");
  const filterNeighborhood = document.getElementById("filter-neighborhood");
  const filterUsage = document.getElementById("filter-usage");
  const filterSize = document.getElementById("filter-size");
  const findSimilarBtn = document.getElementById("find-similar-btn");
  const filterSearchBtn = document.getElementById("filter-search-btn");
  const resultsContainer = document.getElementById("recommendation-results");
  const selectedInfoCard = document.getElementById("selected-household-info");

  // Add each household as an option in the dropdown
  householdData.forEach(function (h) {
    const option = document.createElement("option");
    option.value = h.id;
    option.textContent = h.id + " — " + h.neighborhood + " (" + h.usage + " L, " + h.size + " people)";
    householdSelect.appendChild(option);
  });

  // ========================================================================
  // TAB SWITCHING: Select Household vs. Filter Search
  // ========================================================================
  const tabBtns = document.querySelectorAll(".ml-tab-btn");
  const tabPanels = document.querySelectorAll(".ml-tab-panel");

  tabBtns.forEach(function (btn) {
    btn.addEventListener("click", function () {
      // Remove active state from all tabs
      tabBtns.forEach(function (b) { b.classList.remove("active"); });
      tabPanels.forEach(function (p) { p.classList.remove("active"); });

      // Activate the clicked tab
      btn.classList.add("active");
      var target = btn.getAttribute("data-tab");
      document.getElementById(target).classList.add("active");

      // Clear results when switching tabs
      resultsContainer.innerHTML = "";
      selectedInfoCard.style.display = "none";
    });
  });

  // ========================================================================
  // EVENT: "Find Similar Households" button click
  // ========================================================================
  findSimilarBtn.addEventListener("click", function () {
    var selectedId = householdSelect.value;
    if (!selectedId) {
      resultsContainer.innerHTML = '<p class="ml-no-results">⚠️ Please select a household first.</p>';
      return;
    }

    // Show the selected household's info card
    var selected = householdData.find(function (h) { return h.id === selectedId; });
    showSelectedInfo(selected);

    // Run the ML recommendation engine
    var recommendations = getRecommendations(selectedId, 5);

    // Render the results with animation
    renderRecommendations(recommendations, selected);
  });

  // ========================================================================
  // EVENT: "Search" button click (filter-based)
  // ========================================================================
  filterSearchBtn.addEventListener("click", function () {
    var neighborhood = filterNeighborhood.value;
    var maxUsage = parseFloat(filterUsage.value);
    var size = parseInt(filterSize.value);

    if (!neighborhood || isNaN(maxUsage) || isNaN(size)) {
      resultsContainer.innerHTML = '<p class="ml-no-results">⚠️ Please fill in all filter fields.</p>';
      return;
    }

    // Hide the selected card (not applicable for filter mode)
    selectedInfoCard.style.display = "none";

    // Run the filter-based recommendation
    var recommendations = getFilteredRecommendations(neighborhood, maxUsage, size, 5);

    // Show a virtual "query" info
    showFilterInfo(neighborhood, maxUsage, size);

    // Render results
    renderRecommendations(recommendations, null);
  });

  // ========================================================================
  // RENDER: Display the selected household's details
  // ========================================================================
  function showSelectedInfo(household) {
    selectedInfoCard.style.display = "block";
    selectedInfoCard.innerHTML =
      '<div class="ml-selected-header">' +
        '<span class="ml-selected-badge">📍 Selected Household</span>' +
        '<h4>' + household.id + '</h4>' +
      '</div>' +
      '<div class="ml-selected-stats">' +
        '<div class="ml-selected-stat">' +
          '<span class="ml-stat-value">' + household.neighborhood + '</span>' +
          '<span class="ml-stat-label">Neighborhood</span>' +
        '</div>' +
        '<div class="ml-selected-stat">' +
          '<span class="ml-stat-value">' + household.usage + ' L</span>' +
          '<span class="ml-stat-label">Water Usage</span>' +
        '</div>' +
        '<div class="ml-selected-stat">' +
          '<span class="ml-stat-value">' + household.size + '</span>' +
          '<span class="ml-stat-label">Household Size</span>' +
        '</div>' +
        '<div class="ml-selected-stat">' +
          '<span class="ml-stat-value">$' + household.bill + '</span>' +
          '<span class="ml-stat-label">Monthly Bill</span>' +
        '</div>' +
      '</div>';
  }

  // ========================================================================
  // RENDER: Display filter search info
  // ========================================================================
  function showFilterInfo(neighborhood, usage, size) {
    selectedInfoCard.style.display = "block";
    selectedInfoCard.innerHTML =
      '<div class="ml-selected-header">' +
        '<span class="ml-selected-badge">🔍 Search Criteria</span>' +
        '<h4>Custom Filter Query</h4>' +
      '</div>' +
      '<div class="ml-selected-stats">' +
        '<div class="ml-selected-stat">' +
          '<span class="ml-stat-value">' + neighborhood + '</span>' +
          '<span class="ml-stat-label">Neighborhood</span>' +
        '</div>' +
        '<div class="ml-selected-stat">' +
          '<span class="ml-stat-value">' + usage + ' L</span>' +
          '<span class="ml-stat-label">Max Usage</span>' +
        '</div>' +
        '<div class="ml-selected-stat">' +
          '<span class="ml-stat-value">' + size + '</span>' +
          '<span class="ml-stat-label">Household Size</span>' +
        '</div>' +
      '</div>';
  }

  // ========================================================================
  // RENDER: Display recommendation cards with scores and breakdowns
  // ========================================================================
  function renderRecommendations(recommendations, selected) {
    // Clear previous results
    resultsContainer.innerHTML = "";

    if (recommendations.length === 0) {
      resultsContainer.innerHTML = '<p class="ml-no-results">No similar households found.</p>';
      return;
    }

    // Add a heading
    var heading = document.createElement("h4");
    heading.className = "ml-results-heading";
    heading.textContent = "🤖 Top " + recommendations.length + " Similar Households (by Cosine Similarity)";
    resultsContainer.appendChild(heading);

    // Create a card for each recommendation
    recommendations.forEach(function (rec, index) {
      var card = document.createElement("div");
      card.className = "ml-rec-card";
      // Add staggered animation delay
      card.style.animationDelay = (index * 0.1) + "s";

      // Determine score visual
      var scoreClass = getScoreClass(rec.score);
      var scorePercent = formatScore(rec.score);

      // Build the feature match indicators
      var matchIndicators = "";
      if (rec.featureBreakdown.sameNeighborhood) {
        matchIndicators += '<span class="ml-match-tag match-yes">📍 Same Area</span>';
      }
      if (parseFloat(rec.featureBreakdown.usageDifference) < 50) {
        matchIndicators += '<span class="ml-match-tag match-yes">💧 Similar Usage</span>';
      }
      if (rec.featureBreakdown.sizeDifference <= 1) {
        matchIndicators += '<span class="ml-match-tag match-yes">👨‍👩‍👧 Similar Size</span>';
      }
      if (parseFloat(rec.featureBreakdown.billDifference) < 3) {
        matchIndicators += '<span class="ml-match-tag match-yes">💰 Similar Bill</span>';
      }

      card.innerHTML =
        '<div class="ml-rec-card-header">' +
          '<div class="ml-rec-rank">#' + (index + 1) + '</div>' +
          '<div class="ml-rec-id">' + rec.household.id + '</div>' +
          '<div class="ml-rec-score ' + scoreClass + '">' + scorePercent + ' match</div>' +
        '</div>' +
        '<div class="ml-rec-card-body">' +
          '<div class="ml-rec-detail">' +
            '<span class="ml-detail-icon">🏘️</span>' +
            '<span class="ml-detail-text">' + rec.household.neighborhood + '</span>' +
          '</div>' +
          '<div class="ml-rec-detail">' +
            '<span class="ml-detail-icon">💧</span>' +
            '<span class="ml-detail-text">' + rec.household.usage + ' L/month</span>' +
          '</div>' +
          '<div class="ml-rec-detail">' +
            '<span class="ml-detail-icon">👥</span>' +
            '<span class="ml-detail-text">' + rec.household.size + ' people</span>' +
          '</div>' +
          '<div class="ml-rec-detail">' +
            '<span class="ml-detail-icon">💵</span>' +
            '<span class="ml-detail-text">$' + rec.household.bill + '/month</span>' +
          '</div>' +
        '</div>' +
        '<div class="ml-match-tags">' + matchIndicators + '</div>' +
        '<div class="ml-similarity-bar">' +
          '<div class="ml-similarity-fill ' + scoreClass + '" style="width: ' + scorePercent + '"></div>' +
        '</div>';

      resultsContainer.appendChild(card);
    });

    // Add the "How it works" explanation card at the bottom
    var explainerCard = document.createElement("div");
    explainerCard.className = "ml-explainer-card";
    explainerCard.innerHTML =
      '<h4>🧠 How Does This Work?</h4>' +
      '<div class="ml-explainer-steps">' +
        '<div class="ml-step">' +
          '<div class="ml-step-num">1</div>' +
          '<div class="ml-step-text"><strong>Feature Engineering</strong><br>Each household is converted into a numerical vector using one-hot encoding (neighborhood) and min-max normalization (usage, size, bill).</div>' +
        '</div>' +
        '<div class="ml-step">' +
          '<div class="ml-step-num">2</div>' +
          '<div class="ml-step-text"><strong>Vectorization</strong><br>The selected household becomes an 8-dimensional feature vector: [GP, MS, OT, RS, TH, usage, size, bill].</div>' +
        '</div>' +
        '<div class="ml-step">' +
          '<div class="ml-step-num">3</div>' +
          '<div class="ml-step-text"><strong>Cosine Similarity</strong><br>We compute cos(θ) = (A·B) / (||A|| × ||B||) between the selected household and every other household.</div>' +
        '</div>' +
        '<div class="ml-step">' +
          '<div class="ml-step-num">4</div>' +
          '<div class="ml-step-text"><strong>Ranking</strong><br>Households are sorted by similarity score (highest first) and the top 5 are displayed as recommendations.</div>' +
        '</div>' +
      '</div>';

    resultsContainer.appendChild(explainerCard);
  }

});
