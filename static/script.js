/**
 * ChatBot IA - Multi-chat Application
 * Interface client complète pour chatbot avec RAG
 */

// ===== VARIABLES GLOBALES =====
let sessionId = localStorage.getItem('session_id');
if (!sessionId) {
  sessionId = 'sess_' + Math.random().toString(36).substring(2, 12);
  localStorage.setItem('session_id', sessionId);
}

let userId = "anonymous";
let currentThreadId = null;
let chats = [];
let selectedFiles = [];
let mediaRecorder = null;
let audioChunks = [];
let lastAction = null; // Pour le système de retry
let currentAudioElements = new Map(); // Pour gérer plusieurs éléments audio
let fileUrlCache = new Map(); // Cache pour les URLs des fichiers

// ===== ÉLÉMENTS DOM =====
const chatListElem = $("#chat-list");
const chatHistoryElem = $("#chat-history");
const chatTitleElem = $("#chat-title");
const loadingElem = $("#loading");
const fileInfoElem = $("#file-info");
const recordBtn = $("#record-btn");
const recordStatus = $("#record-status");
const notificationContainer = $("#notification-container");

// ===== UTILITAIRES =====

/**
 * Échappe le HTML pour éviter les injections XSS
 */
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

/**
 * Formate une date ISO en format lisible
 */
function formatDate(isoDate) {
  if (!isoDate) return "";
  const d = new Date(isoDate);
  const now = new Date();
  const diff = now - d;
  const days = Math.floor(diff / (1000 * 60 * 60 * 24));

  if (days === 0) return "Aujourd'hui";
  if (days === 1) return "Hier";
  if (days < 7) return `Il y a ${days} jours`;
  return d.toLocaleDateString('fr-FR');
}

/**
 * Génère un ID unique pour un thread
 */
function generateThreadId() {
  return 'thread_' + Math.random().toString(36).substring(2, 12);
}

/**
 * Génère un ID unique pour un message
 */
function generateMessageId() {
  return 'msg_' + Math.random().toString(36).substring(2, 12);
}

/**
 * Débounce une fonction
 */
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

/**
 * Copie du texte dans le presse-papiers
 */
function copyToClipboard(text) {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(() => {
      showNotification("Copié dans le presse-papiers", 'success');
    }).catch(() => {
      fallbackCopyTextToClipboard(text);
    });
  } else {
    fallbackCopyTextToClipboard(text);
  }
}

/**
 * Méthode de fallback pour la copie
 */
function fallbackCopyTextToClipboard(text) {
  const textArea = document.createElement("textarea");
  textArea.value = text;
  textArea.style.position = "fixed";
  textArea.style.top = "-999999px";
  textArea.style.left = "-999999px";
  document.body.appendChild(textArea);
  textArea.focus();
  textArea.select();
  
  try {
    document.execCommand('copy');
    showNotification("Copié dans le presse-papiers", 'success');
  } catch (err) {
    showNotification("Impossible de copier le texte", 'error');
  }
  
  document.body.removeChild(textArea);
}

/**
 * Formate le nom et la taille des fichiers
 */
function formatFileInfo(file) {
  const size = file.size;
  const sizeStr = size < 1024 * 1024 
    ? `${Math.round(size / 1024)} KB`
    : `${Math.round(size / (1024 * 1024) * 100) / 100} MB`;
  
  const typeIcon = getFileTypeIcon(file.name, file.type);
  return { name: file.name, size: sizeStr, icon: typeIcon };
}

/**
 * Retourne l'icône appropriée selon le type de fichier
 */
function getFileTypeIcon(fileName, mimeType) {
  const ext = fileName.split('.').pop().toLowerCase();
  
  if (mimeType.startsWith('image/')) return 'fa-image';
  if (mimeType.startsWith('audio/')) return 'fa-music';
  if (ext === 'pdf') return 'fa-file-pdf';
  if (ext === 'docx' || ext === 'doc') return 'fa-file-word';
  if (ext === 'xlsx' || ext === 'xls') return 'fa-file-excel';
  if (ext === 'pptx' || ext === 'ppt') return 'fa-file-powerpoint';
  
  return 'fa-file';
}

/**
 * Formate la taille d'un fichier en format lisible
 */
function formatFileSize(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Crée un aperçu pour les fichiers image
 */
function createImagePreview(file) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target.result);
    reader.onerror = () => resolve(null);
    reader.readAsDataURL(file);
  });
}

/**
 * Crée un élément d'affichage de fichier professionnel
 */
async function createFileDisplay(file, messageId) {
  const fileInfo = formatFileInfo(file);
  const fileId = `file-${messageId}-${Math.random().toString(36).substr(2, 9)}`;
  
  let previewContent = '';
  let fileActions = '';

  // Créer un aperçu selon le type de fichier
  if (file.type.startsWith('image/')) {
    const imageSrc = await createImagePreview(file);
    if (imageSrc) {
      previewContent = `
        <div class="file-preview image-preview">
          <img src="${imageSrc}" alt="Aperçu de ${escapeHtml(file.name)}" onclick="openImageModal('${imageSrc}', '${escapeHtml(file.name)}')">
        </div>
      `;
    }
    fileActions = `
      <button class="file-action-btn" onclick="openImageModal('${imageSrc}', '${escapeHtml(file.name)}')" title="Voir en grand">
        <i class="fas fa-expand"></i>
      </button>
    `;
  } else if (file.type.startsWith('audio/')) {
    const audioSrc = URL.createObjectURL(file);
    fileUrlCache.set(fileId, audioSrc);
    
    previewContent = `
      <div class="file-preview audio-preview">
        <div class="audio-waveform">
          <i class="fas fa-music"></i>
          <span class="audio-duration">Audio</span>
        </div>
      </div>
    `;
    fileActions = `
      <button class="file-action-btn audio-play-btn" onclick="toggleAudioPlay('${fileId}', '${audioSrc}')" title="Lire/Pause">
        <i class="fas fa-play"></i>
      </button>
    `;
  } else if (file.name.toLowerCase().endsWith('.pdf')) {
    previewContent = `
      <div class="file-preview document-preview">
        <i class="fas fa-file-pdf pdf-icon"></i>
        <span class="document-pages">PDF</span>
      </div>
    `;
  }

  return `
    <div class="file-attachment professional" data-file-id="${fileId}">
      <div class="file-main-content">
        <div class="file-icon-container">
          <i class="fas ${fileInfo.icon} file-type-icon"></i>
        </div>
        <div class="file-details">
          <div class="file-name" title="${escapeHtml(file.name)}">${escapeHtml(file.name)}</div>
          <div class="file-meta">
            <span class="file-size">${formatFileSize(file.size)}</span>
            <span class="file-type">${file.type.split('/')[1] || 'unknown'}</span>
          </div>
        </div>
        <div class="file-actions">
          ${fileActions}
          <button class="file-action-btn" onclick="downloadFile('${fileId}')" title="Télécharger">
            <i class="fas fa-download"></i>
          </button>
        </div>
      </div>
      ${previewContent}
    </div>
  `;
}

/**
 * Ouvre une image en modal
 */
function openImageModal(imageSrc, fileName) {
  const modal = $(`
    <div class="image-modal-overlay" onclick="closeImageModal()">
      <div class="image-modal-content" onclick="event.stopPropagation()">
        <div class="image-modal-header">
          <h3>${escapeHtml(fileName)}</h3>
          <button class="modal-close" onclick="closeImageModal()">
            <i class="fas fa-times"></i>
          </button>
        </div>
        <div class="image-modal-body">
          <img src="${imageSrc}" alt="${escapeHtml(fileName)}">
        </div>
        <div class="image-modal-actions">
          <button onclick="downloadImageFromModal('${imageSrc}', '${fileName}')" class="btn btn-primary">
            <i class="fas fa-download"></i> Télécharger
          </button>
        </div>
      </div>
    </div>
  `);
  
  $('body').append(modal);
  setTimeout(() => modal.addClass('show'), 10);
}

/**
 * Ferme la modal d'image
 */
function closeImageModal() {
  $('.image-modal-overlay').removeClass('show');
  setTimeout(() => $('.image-modal-overlay').remove(), 300);
}

/**
 * Télécharge une image depuis la modal
 */
function downloadImageFromModal(src, fileName) {
  const a = document.createElement('a');
  a.href = src;
  a.download = fileName;
  a.click();
}

/**
 * Gère la lecture/pause audio
 */
function toggleAudioPlay(fileId, audioSrc) {
  const button = $(`.file-attachment[data-file-id="${fileId}"] .audio-play-btn`);
  const icon = button.find('i');
  
  // Arrêter tous les autres audios
  currentAudioElements.forEach((audio, id) => {
    if (id !== fileId && !audio.paused) {
      audio.pause();
      $(`.file-attachment[data-file-id="${id}"] .audio-play-btn i`).removeClass('fa-pause').addClass('fa-play');
    }
  });

  let audio = currentAudioElements.get(fileId);
  
  if (!audio) {
    audio = new Audio(audioSrc);
    currentAudioElements.set(fileId, audio);
    
    audio.addEventListener('ended', () => {
      icon.removeClass('fa-pause').addClass('fa-play');
    });
    
    audio.addEventListener('error', () => {
      showNotification('Erreur lors de la lecture audio', 'error');
      icon.removeClass('fa-pause').addClass('fa-play');
    });
  }

  if (audio.paused) {
    audio.play().then(() => {
      icon.removeClass('fa-play').addClass('fa-pause');
    }).catch(() => {
      showNotification('Impossible de lire le fichier audio', 'error');
    });
  } else {
    audio.pause();
    icon.removeClass('fa-pause').addClass('fa-play');
  }
}

/**
 * Télécharge un fichier
 */
function downloadFile(fileId) {
  const url = fileUrlCache.get(fileId);
  if (url) {
    const a = document.createElement('a');
    a.href = url;
    a.download = 'file';
    a.click();
  }
}

// ===== SYSTÈME DE NOTIFICATIONS =====

/**
 * Affiche une notification
 */
function showNotification(message, type = 'info', actions = []) {
  const iconMap = {
    success: 'fa-check-circle',
    error: 'fa-exclamation-circle',
    warning: 'fa-exclamation-triangle',
    info: 'fa-info-circle'
  };

  const notification = $(`
    <div class="notification ${type}">
      <i class="fas ${iconMap[type] || iconMap.info} notification-icon"></i>
      <span class="notification-message">${escapeHtml(message)}</span>
      <div class="notification-actions">
        ${actions.map(action => `
          <button class="notification-btn ${action.class || ''}" data-action="${action.id || ''}">
            ${action.icon ? `<i class="${action.icon}"></i>` : ''}
            ${action.text}
          </button>
        `).join('')}
        <button class="notification-btn dismiss-btn">
          <i class="fas fa-times"></i>
          Fermer
        </button>
      </div>
    </div>
  `);
  
  notification.find('[data-action]').on('click', function() {
    const actionId = $(this).data('action');
    const action = actions.find(a => a.id === actionId);
    if (action && action.callback) {
      action.callback();
    }
  });

  notification.find('.dismiss-btn').on('click', () => {
    dismissNotification();
  });

  $('body').append(notification);
  setTimeout(() => notification.addClass('show'), 100);
  
  if (type !== 'error') {
    setTimeout(() => {
      if (notification.hasClass('show')) {
        dismissNotification();
      }
    }, 5000);
  }
}

/**
 * Ferme la notification active
 */
function dismissNotification() {
  const notification = $('.notification.show');
  if (notification.length > 0) {
    notification.removeClass('show');
    setTimeout(() => notification.remove(), 300);
  }
}

// ===== GESTION DES FICHIERS =====

/**
 * Met à jour l'affichage de la liste des fichiers sélectionnés
 */
async function updateFileListDisplay() {
  const fileNameSpan = fileInfoElem.find('.file-name');
  if (selectedFiles.length === 0) {
    fileInfoElem.hide();
    fileNameSpan.html('');
    return;
  }
  
  fileInfoElem.show();
  
  if (selectedFiles.length === 1) {
    const file = selectedFiles[0];
    const fileInfo = formatFileInfo(file);
    let previewHtml = '';
    
    if (file.type.startsWith('image/')) {
      const imageSrc = await createImagePreview(file);
      if (imageSrc) {
        previewHtml = `<img src="${imageSrc}" class="file-mini-preview" alt="Aperçu">`;
      }
    }
    
    fileNameSpan.html(`
      ${previewHtml}
      <i class="fas ${fileInfo.icon}"></i>
      <span class="file-name-text">${escapeHtml(fileInfo.name)}</span>
      <span class="file-size">(${fileInfo.size})</span>
    `);
  } else {
    fileNameSpan.html(`
      <i class="fas fa-files"></i>
      <span class="file-name-text">${selectedFiles.length} fichiers sélectionnés</span>
    `);
  }
}

/**
 * Valide un fichier avant ajout
 */
function validateFile(file) {
  const maxSize = 10 * 1024 * 1024; // 10MB
  const allowedTypes = [
    'application/pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'image/png',
    'image/jpeg',
    'image/jpg',
    'audio/mpeg',
    'audio/wav',
    'audio/mp4',
    'audio/webm'
  ];

  if (file.size > maxSize) {
    showNotification(`Le fichier "${file.name}" est trop volumineux (max 10MB)`, 'error');
    return false;
  }

  if (!allowedTypes.includes(file.type) && !file.name.toLowerCase().match(/\.(pdf|docx|png|jpg|jpeg|mp3|wav|m4a|webm)$/)) {
    showNotification(`Type de fichier non supporté: "${file.name}"`, 'error');
    return false;
  }

  return true;
}

// ===== GESTION DU CHAT =====

/**
 * Charge la liste des conversations
 */
async function loadChatList() {
  try {
    const res = await fetch(`/chats?user_id=${encodeURIComponent(userId)}`);
    if (!res.ok) {
      throw new Error(`Erreur HTTP: ${res.status}`);
    }
    const data = await res.json();
    chats = Array.isArray(data) ? data : [];
    renderChatList();
  } catch (err) {
    console.error("Erreur chargement chats:", err);
    showNotification("Erreur lors du chargement des conversations", 'error');
  }
}

/**
 * Affiche la liste des conversations
 */
function renderChatList() {
  chatListElem.empty();

  if (chats.length === 0) {
    chatListElem.append(`
      <div class="empty-state">
        <i class="fas fa-comments"></i>
        <p>Aucune conversation</p>
        <small>Commencez une nouvelle conversation</small>
      </div>
    `);
    return;
  }

  chats.forEach((chat, index) => {
    const chatElem = $(`
      <div class="chat-list-item" data-thread-id="${chat.thread_id}">
        <div class="chat-item-content">
          <div class="chat-item-header">
            <span class="chat-title" title="${escapeHtml(chat.title)}">${escapeHtml(chat.title)}</span>
            <div class="chat-actions">
              <button class="chat-action-btn edit-btn" title="Renommer">
                <i class="fas fa-edit"></i>
              </button>
              <button class="chat-action-btn delete-btn" title="Supprimer">
                <i class="fas fa-trash"></i>
              </button>
            </div>
          </div>
          <span class="chat-date">${formatDate(chat.created_at)}</span>
          <div class="chat-preview">${escapeHtml(chat.preview || '')}</div>
        </div>
      </div>
    `);

    setTimeout(() => {
      chatElem.addClass('animate-in');
    }, index * 50);

    chatElem.on("click", function(e) {
      if ($(e.target).closest('.chat-actions').length === 0) {
        if (currentThreadId !== chat.thread_id) {
          selectChat(chat.thread_id);
        }
      }
    });

    chatElem.find('.delete-btn').on("click", async (e) => {
      e.stopPropagation();
      if (confirm(`Êtes-vous sûr de vouloir supprimer la conversation "${chat.title}" ?`)) {
        await deleteThread(chat.thread_id);
      }
    });

    chatElem.find('.edit-btn').on("click", (e) => {
      e.stopPropagation();
      startRenameChat(chatElem, chat);
    });

    if (chat.thread_id === currentThreadId) {
      chatElem.addClass("active");
    }

    chatListElem.append(chatElem);
  });
}

/**
 * Sélectionne une conversation
 */
function selectChat(threadId) {
  currentThreadId = threadId;
  renderChatList();
  loadChatHistory(threadId);
  
  const chat = chats.find(c => c.thread_id === threadId);
  if (chat) {
    chatTitleElem.text(chat.title);
  }
}

/**
 * Démarre le renommage d'une conversation
 */
function startRenameChat(chatElem, chat) {
  const titleSpan = chatElem.find('.chat-title');
  const currentTitle = chat.title;

  const input = $(`<input type="text" class="rename-input" value="${escapeHtml(currentTitle)}" maxlength="100">`);
  titleSpan.replaceWith(input);
  input.focus().select();

  const finishRename = async (newTitle) => {
    if (newTitle && newTitle.trim() !== currentTitle) {
      await renameThread(chat.thread_id, newTitle.trim());
    }
    renderChatList();
  };

  input.on('blur', function() {
    finishRename($(this).val());
  });

  input.on('keydown', function(e) {
    if (e.key === 'Enter') {
      e.preventDefault();
      finishRename($(this).val());
    } else if (e.key === 'Escape') {
      renderChatList();
    }
  });
}

/**
 * Charge l'historique d'une conversation
 */
async function loadChatHistory(threadId) {
  if (!threadId) return;
  
  chatHistoryElem.empty();
  showWelcomeMessage("Chargement de la conversation...");
  
  try {
    const res = await fetch(`/history?session_id=${encodeURIComponent(sessionId)}&user_id=${encodeURIComponent(userId)}&thread_id=${encodeURIComponent(threadId)}`);
    if (!res.ok) {
      throw new Error(`Erreur HTTP: ${res.status}`);
    }
    
    const data = await res.json();
    chatHistoryElem.empty();
    
    if (Array.isArray(data) && data.length > 0) {
      for (const msg of data) {
        await addMessage(msg.role, msg.message, msg.context || [], false, msg.files);
      }
    } else {
      showWelcomeMessage();
    }
  } catch (e) {
    console.error("Erreur chargement historique:", e);
    showNotification("Erreur lors du chargement de l'historique", 'error');
    showWelcomeMessage("Erreur de chargement");
  }
}

/**
 * Affiche le message de bienvenue
 */
function showWelcomeMessage(customMessage = null) {
  const message = customMessage || "Posez vos questions ou importez des documents pour commencer une conversation intelligente.";
  chatHistoryElem.html(`
    <div class="welcome-message">
      <div class="welcome-icon">
        <i class="fas fa-comments"></i>
      </div>
      <h3>Bienvenue dans votre assistant IA</h3>
      <p>${message}</p>
    </div>
  `);
}

/**
 * Ajoute un message à l'historique
 */
async function addMessage(role, text, context = [], isLoading = false, files = []) {
  chatHistoryElem.find('.welcome-message').remove();

  const messageId = generateMessageId();
  let processedText = text;

  // Traitement du markdown pour les messages de l'assistant
  if (role === 'assistant' && window.marked && !isLoading) {
    try {
      processedText = marked.parse(text);
    } catch (e) {
      console.error("Erreur parsing markdown:", e);
      processedText = escapeHtml(text);
    }
  } else if (!isLoading) {
    processedText = escapeHtml(text);
  }

  // Création du HTML des fichiers pour les messages utilisateur avec affichage professionnel
  let filesHtml = '';
  if (files && files.length > 0) {
    const fileDisplays = await Promise.all(
      files.map(file => createFileDisplay(file, messageId))
    );
    filesHtml = `
      <div class="message-files professional">
        ${fileDisplays.join('')}
      </div>
    `;
  }

  // Création du HTML des sources pour l'assistant
  const contextHtml = context.length > 0 ? `
    <details class="context-details">
      <summary>
        <i class="fas fa-book"></i>
        Sources utilisées (${context.length})
      </summary>
      <ul class="context-list">
        ${context.map(c => `
          <li class="context-item">
            <div class="context-source">
              <i class="fas fa-file-alt"></i>
              <strong>${escapeHtml(c.metadata?.source || "Document")}</strong>
            </div>
            <div class="context-content">${escapeHtml(c.page_content || c.content || '')}</div>
          </li>
        `).join('')}
      </ul>
    </details>
  ` : '';

  // Actions du message
  const messageActions = !isLoading ? `
    <div class="message-actions">
      ${role === 'user' ? `
        <button class="message-action-btn edit-message-btn" title="Modifier le message" data-message-id="${messageId}">
          <i class="fas fa-edit"></i>
        </button>
      ` : ''}
      <button class="message-action-btn copy-message-btn" title="Copier le message" data-message-id="${messageId}">
        <i class="fas fa-copy"></i>
      </button>
    </div>
  ` : '';

  const loadingIndicator = isLoading ? `
    <div class="typing-indicator">
      <span></span>
      <span></span>
      <span></span>
    </div>
  ` : '';

  const msgDiv = $(`
    <div class="message ${role} ${isLoading ? 'loading' : ''}" data-message-id="${messageId}">
      <div class="message-avatar">
        <i class="fas ${role === 'user' ? 'fa-user' : 'fa-robot'}"></i>
      </div>
      <div class="message-content">
        <div class="message-header">
          <span class="message-role">${role === 'user' ? 'Vous' : 'Assistant IA'}</span>
          <div class="message-time">${new Date().toLocaleTimeString('fr-FR')}</div>
        </div>
        ${filesHtml}
        <div class="message-text" data-original-text="${escapeHtml(text)}">${isLoading ? loadingIndicator : processedText}</div>
        ${role === 'assistant' ? contextHtml : ''}
        ${messageActions}
      </div>
    </div>
  `);

  msgDiv.addClass('animate-in');
  chatHistoryElem.append(msgDiv);
  
  // Scroll automatique si activé
  if (localStorage.getItem('auto-scroll') !== 'false') {
    chatHistoryElem.scrollTop(chatHistoryElem[0].scrollHeight);
  }

  // Retraitement MathJax si nécessaire
  if (window.MathJax && window.MathJax.typesetPromise && !isLoading) {
    MathJax.typesetPromise([msgDiv[0]]).catch(err => {
      console.error("Erreur MathJax:", err);
    });
  }

  return msgDiv;
}

/**
 * Met à jour un message en cours de chargement
 */
function updateLoadingMessage(messageId, newText, context = []) {
  const msgDiv = $(`.message[data-message-id="${messageId}"]`);
  if (msgDiv.length === 0) return;

  msgDiv.removeClass('loading');
  
  // Traitement du markdown
  let processedText = newText;
  if (window.marked) {
    try {
      processedText = marked.parse(newText);
    } catch (e) {
      processedText = escapeHtml(newText);
    }
  }

  // Mise à jour du contenu
  msgDiv.find('.message-text').html(processedText).attr('data-original-text', escapeHtml(newText));
  
  // Ajout du contexte si disponible
  if (context.length > 0) {
    const contextHtml = `
      <details class="context-details">
        <summary>
          <i class="fas fa-book"></i>
          Sources utilisées (${context.length})
        </summary>
        <ul class="context-list">
          ${context.map(c => `
            <li class="context-item">
              <div class="context-source">
                <i class="fas fa-file-alt"></i>
                <strong>${escapeHtml(c.metadata?.source || "Document")}</strong>
              </div>
              <div class="context-content">${escapeHtml(c.page_content || c.content || '')}</div>
            </li>
          `).join('')}
        </ul>
      </details>
    `;
    msgDiv.find('.message-content').append(contextHtml);
  }

  // Ajout des actions
  const messageActions = `
    <div class="message-actions">
      <button class="message-action-btn copy-message-btn" title="Copier le message" data-message-id="${messageId}">
        <i class="fas fa-copy"></i>
      </button>
    </div>
  `;
  msgDiv.find('.message-content').append(messageActions);

  // Retraitement MathJax
  if (window.MathJax && window.MathJax.typesetPromise) {
    MathJax.typesetPromise([msgDiv[0]]).catch(err => {
      console.error("Erreur MathJax:", err);
    });
  }
}

/**
 * Crée une nouvelle conversation
 */
function createNewChat() {
  currentThreadId = null;
  chatTitleElem.text("Nouvelle conversation");
  showWelcomeMessage();
  renderChatList();
  $("#chat-input").focus();
}

/**
 * Envoie un message - Version améliorée avec affichage immédiat
 */
async function sendMessage(e) {
  e.preventDefault();
  
  const question = $("#chat-input").val().trim();
  const useRag = $("#use-rag").is(":checked");

  // Validation
  if (!question && selectedFiles.length === 0) {
    showNotification("Veuillez saisir une question ou importer un fichier", 'warning');
    $("#chat-input").focus();
    return;
  }

  if (!sessionId) {
    showNotification("Erreur de session. Veuillez rafraîchir la page", 'error');
    return;
  }

  // Affichage immédiat du message utilisateur
  const userMessage = question || selectedFiles.map(f => `📎 ${f.name}`).join(', ');
  const filesForDisplay = [...selectedFiles]; // Copie pour l'affichage
  await addMessage("user", userMessage, [], false, filesForDisplay);

  // Affichage immédiat du message de chargement de l'assistant
  const loadingMessageId = generateMessageId();
  const loadingMsg = await addMessage("assistant", "", [], true);
  loadingMsg.attr('data-message-id', loadingMessageId);

  // Préparation des données
  const formData = new FormData();
  formData.append("question", question);
  formData.append("use_rag", useRag);
  formData.append("session_id", sessionId);
  formData.append("user_id", userId);
  formData.append("thread_id", currentThreadId || "");

  for (const file of selectedFiles) {
    formData.append("file", file);
  }

  // Sauvegarde pour retry
  lastAction = { type: 'sendMessage', data: { question, useRag, files: [...selectedFiles] } };

  // Nettoyage de l'interface
  $("#chat-input").val("").css("height", "auto");
  selectedFiles = [];
  updateFileListDisplay();

  // UI Loading
  const sendBtn = $("#send-btn");
  sendBtn.prop("disabled", true).addClass("sending");
  
  try {
    const response = await fetch("/ask", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (data.error) {
      // Suppression du message de chargement en cas d'erreur
      loadingMsg.remove();
      await addMessage("assistant", `❌ Erreur: ${data.error}`, []);
      showNotification(data.error, 'error', [{
        id: 'retry',
        text: 'Réessayer',
        icon: 'fas fa-redo',
        callback: () => {
          // Restaurer les données et renvoyer
          $("#chat-input").val(question);
          selectedFiles = filesForDisplay;
          updateFileListDisplay();
          sendMessage(e);
        }
      }]);
    } else {
      // Mise à jour du message de chargement avec la vraie réponse
      updateLoadingMessage(loadingMessageId, data.answer, data.context || []);

      // Mise à jour du thread
      if (data.thread_id) {
        currentThreadId = data.thread_id;
        await loadChatList();
        
        const chat = chats.find(c => c.thread_id === currentThreadId);
        if (chat) {
          chatTitleElem.text(chat.title);
        }
      }

      // Notification sonore
      if (localStorage.getItem('sound-notifications') !== 'false') {
        playNotificationSound();
      }
    }
  } catch (err) {
    console.error("Erreur envoi message:", err);
    loadingMsg.remove();
    await addMessage("assistant", "❌ Erreur de connexion. Veuillez vérifier votre réseau.");
    showNotification("Erreur de connexion. Vérifiez votre réseau", 'error', [{
      id: 'retry',
      text: 'Réessayer',
      icon: 'fas fa-redo',
      callback: () => sendMessage(e)
    }]);
  } finally {
    sendBtn.prop("disabled", false).removeClass("sending");
  }
}

/**
 * Supprime un thread
 */
async function deleteThread(threadId) {
  try {
    const res = await fetch(`/threads/${encodeURIComponent(threadId)}`, { 
      method: 'DELETE' 
    });
    
    if (!res.ok) {
      throw new Error(`Erreur HTTP: ${res.status}`);
    }

    chats = chats.filter(c => c.thread_id !== threadId);

    if (currentThreadId === threadId) {
      if (chats.length > 0) {
        selectChat(chats[0].thread_id);
      } else {
        createNewChat();
      }
    }

    renderChatList();
    showNotification("Conversation supprimée avec succès", 'success');
  } catch (err) {
    console.error("Erreur suppression thread:", err);
    showNotification("Erreur lors de la suppression", 'error');
  }
}

/**
 * Renomme un thread
 */
async function renameThread(threadId, newTitle) {
  try {
    const res = await fetch(`/threads/${encodeURIComponent(threadId)}`, {
      method: 'PUT',
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: newTitle })
    });
    
    if (!res.ok) {
      throw new Error(`Erreur HTTP: ${res.status}`);
    }

    const chat = chats.find(c => c.thread_id === threadId);
    if (chat) {
      chat.title = newTitle;
      if (currentThreadId === threadId) {
        chatTitleElem.text(newTitle);
      }
    }
    
    showNotification("Conversation renommée avec succès", 'success');
  } catch (err) {
    console.error("Erreur renommage thread:", err);
    showNotification("Erreur lors du renommage", 'error');
  }
}

// ===== GESTION AUDIO =====

/**
 * Démarre l'enregistrement audio
 */
function startRecording() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showNotification("Votre navigateur ne supporte pas l'enregistrement audio", 'error');
    return;
  }

  navigator.mediaDevices.getUserMedia({ 
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      sampleRate: 44100
    }
  }).then(stream => {
    mediaRecorder = new MediaRecorder(stream, {
      mimeType: MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/mp4'
    });
    audioChunks = [];

    mediaRecorder.addEventListener("start", () => {
      recordStatus.show();
      recordBtn.addClass('recording');
      recordBtn.html('<i class="fas fa-stop"></i>');
    });

    mediaRecorder.addEventListener("dataavailable", event => {
      if (event.data.size > 0) {
        audioChunks.push(event.data);
      }
    });

    mediaRecorder.addEventListener("stop", () => {
      recordStatus.hide();
      recordBtn.removeClass('recording');
      recordBtn.html('<i class="fas fa-microphone"></i>');
      
      stream.getTracks().forEach(track => track.stop());
      
      if (audioChunks.length > 0) {
        const audioBlob = new Blob(audioChunks, { 
          type: mediaRecorder.mimeType || 'audio/webm' 
        });
        
        // Créer un fichier temporaire pour l'affichage
        const audioFile = new File([audioBlob], "enregistrement_audio.webm", {
          type: audioBlob.type
        });
        
        selectedFiles = [audioFile];
        updateFileListDisplay();
        sendAudioMessage(audioBlob);
      }
    });

    mediaRecorder.addEventListener("error", (event) => {
      console.error("Erreur MediaRecorder:", event.error);
      showNotification("Erreur lors de l'enregistrement", 'error');
      stopRecording();
    });

    mediaRecorder.start(250);
  }).catch(err => {
    console.error("Erreur accès microphone:", err);
    showNotification("Impossible d'accéder au microphone", 'error');
  });
}

/**
 * Arrête l'enregistrement audio
 */
function stopRecording() {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
  }
}

/**
 * Envoie un message audio
 */
async function sendAudioMessage(audioBlob) {
  if (!sessionId) {
    showNotification("Session non définie", 'error');
    return;
  }

  // Affichage immédiat du message utilisateur avec l'audio
  const audioFile = { name: "🎤 Message audio", size: audioBlob.size, type: "audio/webm" };
  await addMessage("user", "🎤 Message audio envoyé", [], false, [audioFile]);

  // Message de chargement pour l'assistant
  const loadingMessageId = generateMessageId();
  const loadingMsg = await addMessage("assistant", "", [], true);
  loadingMsg.attr('data-message-id', loadingMessageId);

  const formData = new FormData();
  formData.append("question", "");
  formData.append("use_rag", $("#use-rag").is(":checked"));
  formData.append("session_id", sessionId);
  formData.append("user_id", userId);
  formData.append("thread_id", currentThreadId || "");
  formData.append("file", audioBlob, "audio_message.webm");

  const sendBtn = $("#send-btn");
  sendBtn.prop("disabled", true).addClass('sending');

  try {
    const response = await fetch("/ask", {
      method: "POST",
      body: formData,
    });
    
    const data = await response.json();

    if (data.error) {
      loadingMsg.remove();
      await addMessage("assistant", `❌ Erreur: ${data.error}`, []);
      showNotification(data.error, 'error');
    } else {
      updateLoadingMessage(loadingMessageId, data.answer, data.context || []);
      
      if (data.thread_id) {
        currentThreadId = data.thread_id;
        await loadChatList();
      }
    }
  } catch (err) {
    console.error("Erreur envoi audio:", err);
    loadingMsg.remove();
    await addMessage("assistant", "❌ Erreur lors de l'envoi de l'audio");
    showNotification("Erreur lors de l'envoi de l'audio", 'error');
  } finally {
    sendBtn.prop("disabled", false).removeClass('sending');
    selectedFiles = [];
    updateFileListDisplay();
  }
}

// ===== GESTION DES ACTIONS DE MESSAGE =====

/**
 * Édite un message utilisateur et soumet automatiquement la nouvelle question
 */
function editMessage(messageId) {
  const msgDiv = $(`.message[data-message-id="${messageId}"]`);
  const originalText = msgDiv.find('.message-text').attr('data-original-text');
  
  if (!originalText) return;

  const textarea = $(`
    <div class="edit-message-container">
      <textarea class="edit-message-textarea" rows="3" placeholder="Modifiez votre message...">${escapeHtml(originalText)}</textarea>
      <div class="edit-message-actions">
        <button class="btn btn-primary save-edit-btn">
          <i class="fas fa-paper-plane"></i> Envoyer la nouvelle question
        </button>
        <button class="btn btn-secondary cancel-edit-btn">
          <i class="fas fa-times"></i> Annuler
        </button>
      </div>
      <small class="edit-help-text">
        <i class="fas fa-info-circle"></i> 
        En sauvegardant, l'assistant répondra à votre nouvelle question.
      </small>
    </div>
  `);

  const messageText = msgDiv.find('.message-text');
  const messageActions = msgDiv.find('.message-actions');
  
  messageText.hide();
  messageActions.hide();
  messageText.after(textarea);

  // Auto-resize du textarea
  const textareaEl = textarea.find('.edit-message-textarea')[0];
  textareaEl.style.height = 'auto';
  textareaEl.style.height = Math.min(textareaEl.scrollHeight, 200) + 'px';
  textareaEl.focus();
  textareaEl.setSelectionRange(textareaEl.value.length, textareaEl.value.length);

  // Fonction pour annuler l'édition
  const cancelEdit = () => {
    textarea.remove();
    messageText.show();
    messageActions.show();
  };

  // Fonction pour sauvegarder et envoyer la nouvelle question
  const saveAndSend = async () => {
    const newText = textarea.find('.edit-message-textarea').val().trim();
    
    if (!newText) {
      showNotification("La question ne peut pas être vide", 'warning');
      return;
    }
    
    if (newText === originalText) {
      cancelEdit();
      return;
    }

    // Mise à jour visuelle du message
    messageText.text(newText).attr('data-original-text', newText);
    cancelEdit();

    // Supprimer tous les messages suivants (assistant et utilisateur)
    const allMessages = $("#chat-history .message");
    const currentIndex = allMessages.index(msgDiv);
    
    // Supprimer tous les messages après celui-ci
    allMessages.slice(currentIndex + 1).each(function() {
      // Nettoyer les URLs d'objets pour éviter les fuites mémoire
      const messageId = $(this).data('message-id');
      if (messageId) {
        const url = fileUrlCache.get(messageId);
        if (url) {
          URL.revokeObjectURL(url);
          fileUrlCache.delete(messageId);
        }
      }
      $(this).remove();
    });

    // Envoyer la nouvelle question
    await sendEditedMessage(newText);
  };

  // Gestion des boutons
  textarea.find('.save-edit-btn').on('click', saveAndSend);
  textarea.find('.cancel-edit-btn').on('click', cancelEdit);

  // Auto-resize pendant la saisie
  textarea.find('.edit-message-textarea').on('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 200) + 'px';
  });

  // Raccourcis clavier
  textarea.find('.edit-message-textarea').on('keydown', function(e) {
    if (e.ctrlKey && e.key === 'Enter') {
      e.preventDefault();
      saveAndSend();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      cancelEdit();
    }
  });
}

/**
 * Envoie une question éditée
 */
async function sendEditedMessage(question) {
  if (!sessionId) {
    showNotification("Erreur de session. Veuillez rafraîchir la page", 'error');
    return;
  }

  const useRag = $("#use-rag").is(":checked");

  // Affichage du message de chargement de l'assistant
  const loadingMessageId = generateMessageId();
  const loadingMsg = await addMessage("assistant", "", [], true);
  loadingMsg.attr('data-message-id', loadingMessageId);

  // Préparation des données
  const formData = new FormData();
  formData.append("question", question);
  formData.append("use_rag", useRag);
  formData.append("session_id", sessionId);
  formData.append("user_id", userId);
  formData.append("thread_id", currentThreadId || "");

  // Sauvegarde pour retry
  lastAction = { type: 'sendEditedMessage', data: { question, useRag } };

  // UI Loading
  const sendBtn = $("#send-btn");
  sendBtn.prop("disabled", true).addClass("sending");
  
  try {
    const response = await fetch("/ask", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (data.error) {
      // Suppression du message de chargement en cas d'erreur
      loadingMsg.remove();
      await addMessage("assistant", `❌ Erreur: ${data.error}`, []);
      showNotification(data.error, 'error', [{
        id: 'retry',
        text: 'Réessayer',
        icon: 'fas fa-redo',
        callback: () => sendEditedMessage(question)
      }]);
    } else {
      // Mise à jour du message de chargement avec la vraie réponse
      updateLoadingMessage(loadingMessageId, data.answer, data.context || []);

      // Mise à jour du thread
      if (data.thread_id) {
        currentThreadId = data.thread_id;
        await loadChatList();
        
        const chat = chats.find(c => c.thread_id === currentThreadId);
        if (chat) {
          chatTitleElem.text(chat.title);
        }
      }

      // Notification sonore
      if (localStorage.getItem('sound-notifications') !== 'false') {
        playNotificationSound();
      }

      showNotification("Question modifiée et envoyée avec succès", 'success');
    }
  } catch (err) {
    console.error("Erreur envoi message édité:", err);
    loadingMsg.remove();
    await addMessage("assistant", "❌ Erreur de connexion. Veuillez vérifier votre réseau.");
    showNotification("Erreur de connexion. Vérifiez votre réseau", 'error', [{
      id: 'retry',
      text: 'Réessayer',
      icon: 'fas fa-redo',
      callback: () => sendEditedMessage(question)
    }]);
  } finally {
    sendBtn.prop("disabled", false).removeClass("sending");
  }
}

/**
 * Copie le contenu d'un message
 */
function copyMessage(messageId) {
  const msgDiv = $(`.message[data-message-id="${messageId}"]`);
  const originalText = msgDiv.find('.message-text').attr('data-original-text');
  const textToCopy = originalText || msgDiv.find('.message-text').text().trim();
  
  copyToClipboard(textToCopy);
}

// ===== EXPORT/IMPORT =====

/**
 * Exporte l'historique de conversation
 */
function exportChatHistory(format = "txt") {
  const messages = [];
  const chatTitle = chatTitleElem.text();
  
  if (format === "txt") {
    messages.push(`=== ${chatTitle} ===`);
    messages.push(`Exporté le: ${new Date().toLocaleString('fr-FR')}`);
    messages.push("=".repeat(50));
    messages.push("");

    $("#chat-history .message:not(.loading)").each(function () {
      const role = $(this).hasClass("user") ? "Utilisateur" : "Assistant";
      const originalText = $(this).find(".message-text").attr('data-original-text');
      const text = originalText || $(this).find(".message-text").text().trim();
      const time = $(this).find(".message-time").text().trim();
      
      messages.push(`[${time}] ${role}:`);
      messages.push(text);
      
      // Ajouter contexte s'il existe
      const contextItems = $(this).find(".context-list .context-item");
      if (contextItems.length > 0) {
        messages.push("\nSources utilisées:");
        contextItems.each(function () {
          const source = $(this).find(".context-source").text().trim();
          const content = $(this).find(".context-content").text().trim().substring(0, 100) + "...";
          messages.push(`  • ${source}: ${content}`);
        });
      }
      
      messages.push("");
    });
  } else if (format === "json") {
    const exportData = {
      title: chatTitle,
      exported_at: new Date().toISOString(),
      thread_id: currentThreadId,
      messages: []
    };

    $("#chat-history .message:not(.loading)").each(function () {
      const role = $(this).hasClass("user") ? "user" : "assistant";
      const originalText = $(this).find(".message-text").attr('data-original-text');
      const text = originalText || $(this).find(".message-text").text().trim();
      const time = $(this).find(".message-time").text().trim();
      
      const messageData = { role, content: text, timestamp: time };
      
      if (role === "assistant") {
        const context = [];
        $(this).find(".context-list .context-item").each(function () {
          context.push({
            source: $(this).find(".context-source").text().trim(),
            content: $(this).find(".context-content").text().trim()
          });
        });
        if (context.length > 0) {
          messageData.context = context;
        }
      }
      
      exportData.messages.push(messageData);
    });

    messages.push(JSON.stringify(exportData, null, 2));
  }

  const content = messages.join("\n");
  const blob = new Blob([content], { 
    type: format === "json" ? "application/json" : "text/plain",
    charset: "utf-8" 
  });
  
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${chatTitle.replace(/[^a-z0-9]/gi, '_')}_${new Date().toISOString().slice(0,19).replace(/[:T]/g, "-")}.${format}`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ===== RECHERCHE ET FILTRAGE =====

/**
 * Filtre les conversations par recherche
 */
const searchChats = debounce(function(searchTerm) {
  const search = searchTerm.toLowerCase();
  $('#chat-list .chat-list-item').each(function () {
    const title = $(this).find('.chat-title').text().toLowerCase();
    const preview = $(this).find('.chat-preview').text().toLowerCase();
    const visible = title.includes(search) || preview.includes(search);
    $(this).toggle(visible);
  });
}, 300);

// ===== THÈMES ET PARAMÈTRES =====

/**
 * Applique un thème
 */
function applyTheme(theme) {
  const body = document.body;
  
  if (theme === 'auto') {
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    theme = prefersDark ? 'dark' : 'light';
  }
  
  body.classList.toggle('dark-mode', theme === 'dark');
  localStorage.setItem('theme', theme);
}

/**
 * Joue un son de notification
 */
function playNotificationSound() {
  try {
    const audio = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvGEcBj+a2/LDciUFL4PO8tiJNwgZaLvt559NEAxPqOPwtmMcBjiR1/LNeSsFJHfH8N2QQAoUXrTp66hVFA==');
    audio.volume = 0.3;
    audio.play().catch(() => {});
  } catch (e) {
    // Son non disponible, ignorer silencieusement
  }
}

// ===== GESTION DES MODALES =====

/**
 * Ouvre une modale
 */
function openModal(modalId) {
  const modal = $(`#${modalId}`);
  modal.addClass('show').attr('aria-hidden', 'false');
  
  const focusable = modal.find('input, select, button').first();
  if (focusable.length) {
    setTimeout(() => focusable.focus(), 100);
  }
}

/**
 * Ferme toutes les modales
 */
function closeModals() {
  $('.modal').removeClass('show').attr('aria-hidden', 'true');
}

// ===== NETTOYAGE DES RESSOURCES =====

/**
 * Nettoie les ressources lors du changement de conversation
 */
function cleanupResources() {
  // Arrêter tous les audios en cours
  currentAudioElements.forEach(audio => {
    if (!audio.paused) {
      audio.pause();
    }
  });
  currentAudioElements.clear();

  // Libérer les URLs d'objets
  fileUrlCache.forEach(url => {
    URL.revokeObjectURL(url);
  });
  fileUrlCache.clear();
}

// ===== ÉVÉNEMENTS DOM =====

$(document).ready(function () {
  // === INITIALISATION ===
  
  const savedTheme = localStorage.getItem('theme') || 'light';
  $('#theme-select').val(savedTheme);
  applyTheme(savedTheme);
  
  const soundEnabled = localStorage.getItem('sound-notifications') !== 'false';
  $('#sound-notifications').prop('checked', soundEnabled);
  
  const autoScroll = localStorage.getItem('auto-scroll') !== 'false';
  $('#auto-scroll').prop('checked', autoScroll);

  loadChatList().then(() => {
    if (chats.length > 0) {
      selectChat(chats[0].thread_id);
    } else {
      createNewChat();
    }
  });

  // === ÉVÉNEMENTS DE BASE ===
  
  $("#new-chat-btn").on("click", () => {
    cleanupResources();
    createNewChat();
  });
  
  $("#chat-form").on("submit", sendMessage);
  
  // Gestion améliorée du textarea
  $('#chat-input').on('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 150) + 'px';
  });

  $('#chat-input').on('keydown', function(e) {
    if (e.ctrlKey && e.key === 'Enter') {
      e.preventDefault();
      $('#chat-form').submit();
    }
  });

  $(document).on('keydown', function(e) {
    if (e.key === 'Escape') {
      closeModals();
      closeImageModal();
    }
  });

  // === GESTION DES FICHIERS ===
  
  $('#file-input').on('change', function (e) {
    const newFiles = Array.from(e.target.files);
    let addedCount = 0;
    
    for (const file of newFiles) {
      if (validateFile(file)) {
        if (!selectedFiles.some(existing => 
          existing.name === file.name && 
          existing.size === file.size && 
          existing.lastModified === file.lastModified
        )) {
          selectedFiles.push(file);
          addedCount++;
        }
      }
    }
    
    if (addedCount > 0) {
      updateFileListDisplay();
      showNotification(`${addedCount} fichier(s) ajouté(s)`, 'success');
    }
    
    $(this).val('');
  });

  // Gestion dynamique du bouton de suppression de fichier
  $(document).on('click', '.remove-file-btn', function () {
    selectedFiles = [];
    updateFileListDisplay();
    showNotification('Fichiers supprimés', 'info');
  });

  // === ACTIONS DE MESSAGE ===
  
  $(document).on('click', '.edit-message-btn', function() {
    const messageId = $(this).data('message-id');
    editMessage(messageId);
  });

  $(document).on('click', '.copy-message-btn', function() {
    const messageId = $(this).data('message-id');
    copyMessage(messageId);
  });

  // === ENREGISTREMENT AUDIO ===
  
  recordBtn.on('click', function() {
    if (mediaRecorder && mediaRecorder.state === "recording") {
      stopRecording();
    } else {
      startRecording();
    }
  });

  // === RECHERCHE ===
  
  $('#thread-search-input').on('input', function () {
    searchChats($(this).val());
  });

  // === ACTIONS D'EN-TÊTE ===
  
  $("#settings-btn").on("click", () => openModal('settings-modal'));
  $("#info-btn").on("click", () => openModal('info-modal'));
  
  $("#toggle-context-btn").on("click", function () {
    const details = $("details.context-details");
    const allOpen = details.length > 0 && details.get().every(d => d.open);
    
    details.each(function () {
      this.open = !allOpen;
    });
    
    const icon = $(this).find("i");
    icon.toggleClass("fa-book-open fa-book");
  });
  
  $("#export-btn").on("click", function() {
    exportChatHistory('txt');
    showNotification("Conversation exportée avec succès", "success");
  });

  // === MODALES ===
  
  $('.modal-close').on('click', closeModals);
  $('.modal').on('click', function(e) {
    if (e.target === this) {
      closeModals();
    }
  });

  // === PARAMÈTRES ===
  
  $('#theme-select').on('change', function() {
    applyTheme($(this).val());
  });
  
  $('#sound-notifications').on('change', function() {
    localStorage.setItem('sound-notifications', $(this).is(':checked'));
  });
  
  $('#auto-scroll').on('change', function() {
    localStorage.setItem('auto-scroll', $(this).is(':checked'));
  });

  // === NOTIFICATIONS ===
  
  $(document).on('click', '#dismiss-btn', dismissNotification);
  
  $(document).on('click', '#retry-btn', function() {
    if (lastAction) {
      dismissNotification();
      if (lastAction.type === 'sendMessage') {
        $("#chat-input").val(lastAction.data.question);
        selectedFiles = lastAction.data.files || [];
        updateFileListDisplay();
        $("#use-rag").prop('checked', lastAction.data.useRag);
        $('#chat-form').submit();
      }
    }
  });

  // === DÉTECTION DE CHANGEMENT DE PRÉFÉRENCE SYSTÈME ===
  
  if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(e) {
      if (localStorage.getItem('theme') === 'auto') {
        applyTheme('auto');
      }
    });
  }

  console.log("ChatBot IA - Application améliorée initialisée");
});