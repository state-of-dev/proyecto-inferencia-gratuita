import { handleChat } from '../scripts/server.mjs';

export default function handler(req, res) {
  return handleChat(req, res);
}
